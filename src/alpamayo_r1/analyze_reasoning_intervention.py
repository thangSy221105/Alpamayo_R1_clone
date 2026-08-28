"""Analyze Alpamayo-R1 reasoning intervention JSONL results.

The script is intentionally independent of the model and can analyze a partial
JSONL file while an evaluation is still running. It writes tables, plots, and
human-readable standout examples to one output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


MODES = ("no_reasoning", "noisy", "cross_scene", "opposite_action")
BASELINE_BINS = ((0.0, 0.2, "[0,0.2)"), (0.2, 0.8, "[0.2,0.8)"), (0.8, 1.0000001, "[0.8,1]"))


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            score = record.get("lm_judgement_guided_action", {}).get("consistency_score")
            record["_score"] = as_float(score)
            record["_alpha"] = as_float(record.get("alpha"))
            record["_control_delta"] = as_float(record.get("mean_l2_control_delta_u1_u2"))
            records.append(record)
        except (json.JSONDecodeError, TypeError):
            invalid += 1
    if invalid:
        print(f"Warning: ignored {invalid} invalid JSONL line(s).")
    return records


def choose_baselines(records: list[dict[str, Any]]) -> dict[str, float]:
    """Use no_reasoning/alpha=0 when available, then any alpha=0 record."""
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["_score"] is not None and record["_alpha"] == 0.0:
            candidates[str(record.get("clip_id"))].append(record)

    baselines: dict[str, float] = {}
    for clip_id, values in candidates.items():
        preferred = [r for r in values if r.get("mode") == "no_reasoning"]
        selected = preferred or values
        baselines[clip_id] = float(selected[0]["_score"])
    return baselines


def clip_sensitivities(records: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        delta = record["_control_delta"]
        if delta is not None:
            values[str(record.get("clip_id"))].append(delta)
    return {clip_id: mean(deltas) for clip_id, deltas in values.items() if deltas}


def safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def safe_median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def safe_std(values: list[float]) -> float | None:
    return round(pstdev(values), 6) if len(values) > 1 else 0.0 if values else None


def classify_change(delta: float) -> str:
    if delta <= 0.0:
        return "no_improvement"
    if delta < 0.2:
        return "improve_<0.2"
    if delta <= 0.5:
        return "improve_0.2_to_0.5"
    return "improve_>0.5"


def classify_decrease(delta: float) -> str:
    if delta >= 0.0:
        return "no_decrease"
    if delta > -0.2:
        return "decrease_<0.2"
    if delta >= -0.5:
        return "decrease_0.2_to_0.5"
    return "decrease_>0.5"


def baseline_bin(score: float) -> str:
    for low, high, label in BASELINE_BINS:
        if low <= score < high:
            return label
    return "out_of_range"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(values: list[dict[str, Any]], mode: str, alpha: float) -> dict[str, Any]:
    scores = [r["_score"] for r in values if r["_score"] is not None]
    deltas = [r["_delta"] for r in values if r["_delta"] is not None]
    controls = [r["_control_delta"] for r in values if r["_control_delta"] is not None]
    changes = Counter(classify_change(d) for d in deltas)
    decreases = Counter(classify_decrease(d) for d in deltas)
    return {
        "mode": mode,
        "alpha": alpha,
        "num_records": len(values),
        "num_scored": len(scores),
        "baseline_mean": safe_mean([r["_baseline"] for r in values]),
        "score_mean": safe_mean(scores),
        "score_median": safe_median(scores),
        "score_std": safe_std(scores),
        "delta_mean": safe_mean(deltas),
        "delta_median": safe_median(deltas),
        "delta_std": safe_std(deltas),
        "mean_control_delta": safe_mean(controls),
        "improve_rate": round(sum(d > 0 for d in deltas) / len(deltas), 6) if deltas else None,
        "worsen_rate": round(sum(d < 0 for d in deltas) / len(deltas), 6) if deltas else None,
        "no_improvement": changes["no_improvement"],
        "improve_<0.2": changes["improve_<0.2"],
        "improve_0.2_to_0.5": changes["improve_0.2_to_0.5"],
        "improve_>0.5": changes["improve_>0.5"],
        "no_decrease": decreases["no_decrease"],
        "decrease_<0.2": decreases["decrease_<0.2"],
        "decrease_0.2_to_0.5": decreases["decrease_0.2_to_0.5"],
        "decrease_>0.5": decreases["decrease_>0.5"],
    }


def build_analysis(records: list[dict[str, Any]], baseline: dict[str, float], sensitivity: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for record in records:
        clip_id = str(record.get("clip_id"))
        if record["_score"] is None or clip_id not in baseline:
            continue
        row = dict(record)
        row["_baseline"] = baseline[clip_id]
        row["_delta"] = float(record["_score"]) - baseline[clip_id]
        row["_baseline_bin"] = baseline_bin(baseline[clip_id])
        row["_sensitivity"] = sensitivity.get(clip_id)
        scored.append(row)

    cutoff_values = list(sensitivity.values())
    sensitivity_cutoff = median(cutoff_values) if cutoff_values else None
    for row in scored:
        score = row["_baseline"]
        sensitivity_value = row["_sensitivity"]
        if score < 0.5:
            baseline_class = "low"
        elif score >= 0.75:
            baseline_class = "high"
        else:
            baseline_class = "middle"
        if sensitivity_cutoff is None or sensitivity_value is None:
            sensitivity_class = "unknown"
        else:
            sensitivity_class = "high" if sensitivity_value >= sensitivity_cutoff else "low"
        row["baseline_class"] = baseline_class
        row["sensitivity_class"] = sensitivity_class
        row["four_group"] = (
            f"baseline_{baseline_class}__sensitivity_{sensitivity_class}"
            if baseline_class in {"low", "high"} and sensitivity_class in {"low", "high"}
            else "middle_or_unknown"
        )

    by_config: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_config[(str(row.get("mode")), float(row.get("_alpha")))].append(row)
    config_rows = [metric_row(values, mode, alpha) for (mode, alpha), values in sorted(by_config.items())]
    metadata = {
        "num_input_records": len(records),
        "num_scored_records": len(scored),
        "num_clips_with_baseline": len(baseline),
        "num_clips_with_sensitivity": len(sensitivity),
        "sensitivity_definition": "mean mean_l2_control_delta_u1_u2 across available modes per clip",
        "sensitivity_cutoff": round(sensitivity_cutoff, 6) if sensitivity_cutoff is not None else None,
        "baseline_rule": "low < 0.50; middle [0.50,0.75); high >= 0.75",
        "baseline_bins": [label for _, _, label in BASELINE_BINS],
    }
    return scored, {"metadata": metadata, "config_rows": config_rows}


def write_standouts(path: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    candidates = [r for r in rows if float(r.get("_alpha", 0.0)) != 0.0]
    best = sorted(candidates, key=lambda r: float(r["_delta"]), reverse=True)[:top_k]
    worst = sorted(candidates, key=lambda r: float(r["_delta"]))[:top_k]
    with path.open("w", encoding="utf-8") as file:
        for title, selected in (("TOP IMPROVEMENTS", best), ("TOP DEGRADATIONS", worst)):
            file.write(f"\n{'=' * 90}\n{title}\n")
            for index, row in enumerate(selected, 1):
                judge = row.get("lm_judgement_guided_action", {})
                file.write(
                    f"\n{index}. clip={row.get('clip_id')} mode={row.get('mode')} alpha={row.get('_alpha')}\n"
                    f"baseline={row.get('_baseline'):.4f} score={row.get('_score'):.4f} delta={row.get('_delta'):+.4f}\n"
                    f"clean_reasoning={row.get('clean_reasoning')}\n"
                    f"perturbed_reasoning={row.get('perturbed_reasoning')}\n"
                    f"control_delta={row.get('_control_delta')} guidance_change={row.get('mean_l2_guidance_change_u1')}\n"
                    f"label={judge.get('label')} intended_action={judge.get('intended_action')}\n"
                    f"evidence={judge.get('trajectory_evidence')}\n"
                )


def make_plots(output_dir: Path, scored: list[dict[str, Any]], config_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; tables were written but plots were skipped.")
        return

    plots_dir = output_dir / "plots"
    per_config_dir = plots_dir / "by_config"
    per_config_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['mode']}\na={r['alpha']}" for r in config_rows]
    scores = [r["score_mean"] if r["score_mean"] is not None else 0 for r in config_rows]
    deltas = [r["delta_mean"] if r["delta_mean"] is not None else 0 for r in config_rows]

    def save_bar(filename: str, values: list[float], title: str, ylabel: str, colors: list[str] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.5))
        ax.bar(range(len(values)), values, color=colors or "#4c78a8")
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)

    save_bar("mean_score_by_configuration.png", scores, "Mean consistency score by configuration", "Mean LM score")
    save_bar("mean_delta_by_configuration.png", deltas, "Mean score change from alpha=0", "Mean score delta", ["#59a14f" if x >= 0 else "#e15759" for x in deltas])

    unique_groups = {}
    for row in scored:
        group = row["four_group"]
        if group != "middle_or_unknown":
            unique_groups[str(row["clip_id"])] = group
    group_counts = Counter(unique_groups.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    group_labels = list(group_counts)
    ax.bar(range(len(group_labels)), [group_counts[x] for x in group_labels], color="#4c78a8")
    ax.set_xticks(range(len(group_labels)), [x.replace("__", "\n") for x in group_labels], rotation=20)
    ax.set_ylabel("Number of clip records")
    ax.set_title("Four baseline-sensitivity groups")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "four_group_counts.png", dpi=180)
    plt.close(fig)

    group_delta_rows = []
    for group in sorted(group_counts):
        values = [r["_delta"] for r in scored if r["four_group"] == group and r["_alpha"] != 0]
        group_delta_rows.append((group, mean(values) if values else 0.0))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(group_delta_rows)), [x[1] for x in group_delta_rows], color=["#59a14f" if x[1] >= 0 else "#e15759" for x in group_delta_rows])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(group_delta_rows)), [x[0].replace("__", "\n") for x in group_delta_rows], rotation=20)
    ax.set_ylabel("Mean score delta")
    ax.set_title("Mean intervention effect by group")
    fig.tight_layout()
    fig.savefig(plots_dir / "four_group_mean_delta.png", dpi=180)
    plt.close(fig)

    bin_rows = []
    for bin_label in [x[2] for x in BASELINE_BINS]:
        values = [r["_delta"] for r in scored if r["_baseline_bin"] == bin_label and r["_alpha"] != 0]
        bin_rows.append((bin_label, mean(values) if values else 0.0, len(values)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(bin_rows)), [x[1] for x in bin_rows], color=["#59a14f" if x[1] >= 0 else "#e15759" for x in bin_rows])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(bin_rows)), [x[0] for x in bin_rows])
    ax.set_ylabel("Mean score delta")
    ax.set_title("Intervention effect by baseline-score interval")
    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_bins_mean_delta.png", dpi=180)
    plt.close(fig)

    buckets = ("no_improvement", "improve_<0.2", "improve_0.2_to_0.5", "improve_>0.5", "decrease_<0.2", "decrease_0.2_to_0.5", "decrease_>0.5")
    totals = Counter()
    for row in scored:
        if row["_alpha"] == 0:
            continue
        totals[classify_change(row["_delta"])] += 1
        totals[classify_decrease(row["_delta"])] += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(buckets)), [totals[x] for x in buckets], color=["#bab0ab", "#59a14f", "#76b7b2", "#2f7f5f", "#f28e2b", "#e15759", "#b22222"])
    ax.set_xticks(range(len(buckets)), buckets, rotation=45, ha="right")
    ax.set_ylabel("Number of clip records")
    ax.set_title("Improvement and degradation buckets")
    fig.tight_layout()
    fig.savefig(plots_dir / "change_buckets.png", dpi=180)
    plt.close(fig)

    by_config: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_config[(str(row["mode"]), float(row["_alpha"]))].append(row)
    for (mode, alpha), values in sorted(by_config.items()):
        if alpha == 0:
            continue
        changes = [r["_delta"] for r in values]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(changes, bins=min(15, max(3, len(changes))), color="#4c78a8", edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{mode} | alpha={alpha}")
        ax.set_xlabel("Score delta from alpha=0")
        ax.set_ylabel("Number of clips")
        fig.tight_layout()
        filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{mode}__alpha_{alpha}.png")
        fig.savefig(per_config_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and visualize Alpamayo intervention results.")
    parser.add_argument("--input", default="reasoning_intervention_300_waypoints.jsonl")
    parser.add_argument("--output-dir", default="intervention_analysis")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(input_path)
    baseline = choose_baselines(records)
    sensitivity = clip_sensitivities(records)
    scored, analysis = build_analysis(records, baseline, sensitivity)

    clip_rows = []
    for row in scored:
        clip_rows.append({
            "clip_id": row.get("clip_id"),
            "mode": row.get("mode"),
            "alpha": row.get("_alpha"),
            "baseline_score": row.get("_baseline"),
            "score": row.get("_score"),
            "score_delta": row.get("_delta"),
            "baseline_bin": row.get("_baseline_bin"),
            "baseline_class": row.get("baseline_class"),
            "sensitivity": row.get("_sensitivity"),
            "sensitivity_class": row.get("sensitivity_class"),
            "four_group": row.get("four_group"),
            "control_delta": row.get("_control_delta"),
            "guidance_change": row.get("mean_l2_guidance_change_u1"),
            "label": row.get("lm_judgement_guided_action", {}).get("label"),
        })
    write_csv(output_dir / "clip_metrics.csv", clip_rows)
    write_csv(output_dir / "configuration_metrics.csv", analysis["config_rows"])

    group_rows = []
    for group in sorted({r["four_group"] for r in scored if r["four_group"] != "middle_or_unknown"}):
        values = [r for r in scored if r["four_group"] == group and r["_alpha"] != 0]
        deltas = [r["_delta"] for r in values]
        group_rows.append({
            "four_group": group,
            "num_records": len(values),
            "num_clips": len({r["clip_id"] for r in values}),
            "mean_baseline": safe_mean([r["_baseline"] for r in values]),
            "mean_delta": safe_mean(deltas),
            "median_delta": safe_median(deltas),
            "improve_rate": round(sum(d > 0 for d in deltas) / len(deltas), 6) if deltas else None,
            "worsen_rate": round(sum(d < 0 for d in deltas) / len(deltas), 6) if deltas else None,
        })
    write_csv(output_dir / "four_group_metrics.csv", group_rows)

    bin_rows = []
    for _, _, label in BASELINE_BINS:
        values = [r for r in scored if r["_baseline_bin"] == label and r["_alpha"] != 0]
        deltas = [r["_delta"] for r in values]
        bin_rows.append({
            "baseline_bin": label,
            "num_records": len(values),
            "num_clips": len({r["clip_id"] for r in values}),
            "mean_delta": safe_mean(deltas),
            "median_delta": safe_median(deltas),
            "improve_rate": round(sum(d > 0 for d in deltas) / len(deltas), 6) if deltas else None,
            "worsen_rate": round(sum(d < 0 for d in deltas) / len(deltas), 6) if deltas else None,
        })
    write_csv(output_dir / "baseline_bin_metrics.csv", bin_rows)

    bucket_rows = []
    for config in analysis["config_rows"]:
        bucket_rows.append({"mode": config["mode"], "alpha": config["alpha"], **{key: config[key] for key in (
            "no_improvement", "improve_<0.2", "improve_0.2_to_0.5", "improve_>0.5",
            "no_decrease", "decrease_<0.2", "decrease_0.2_to_0.5", "decrease_>0.5",
        )}})
    write_csv(output_dir / "change_buckets.csv", bucket_rows)

    write_standouts(output_dir / "standout_reasoning.txt", scored, args.top_k)
    analysis["metadata"]["input"] = str(input_path)
    analysis["metadata"]["output_dir"] = str(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(output_dir, scored, analysis["config_rows"])

    print(f"Analyzed {len(records)} input records; {len(scored)} scored records.")
    print(f"Wrote tables, summary, plots, and standout reasoning to: {output_dir}")


if __name__ == "__main__":
    main()
