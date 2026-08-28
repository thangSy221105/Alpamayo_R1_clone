> [!IMPORTANT]
> ## 🚀 [Newer Versions of Alpamayo Are Available](https://github.com/NVlabs/alpamayo-recipes)
>
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on newer Alpamayo versions.
>
> 👉 Visit the new Alpamayo hub: https://github.com/NVlabs/alpamayo-recipes
>
> There you will find the latest Alpamayo models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Alpamayo 1. We encourage all users to migrate to newer versions for the latest state-of-the-art Physical AI capabilities.

<div align="center">

# 🏔️ Alpamayo 1

### Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

## Updates

- [May 2026] Fine-tuning and post-training scripts moved to [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes): [SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_sft) and [RL](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl).
- [April 2026] ⚙️ [Fine-tuning scripts](#fine-tuning-scripts) released: [SFT](docs/FINETUNE_SFT.md) for supervised fine-tuning and [RL](finetune/rl/README.md) for reinforcement learning-based post-training.
- [March 2026] [🏔️ Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) has been released! We recommend all users check out the new version for improved performance, new features, and continued support! 🚀
- [January 2026] Following the release of [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) at CES 2026, Alpamayo-R1 has been renamed to Alpamayo 1.

______________________________________________________________________

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Support

📣 **Usage questions and discussion about Alpamayo 1**: please join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug report, Documentation request, or Feature request). The relevant NVIDIA responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly here.

## Requirements

| Requirement | Specification                                                       |
| ----------- | ------------------------------------------------------------------- |
| **Python**  | 3.12.x (see `pyproject.toml`)                                       |
| **GPU**     | NVIDIA GPU with ≥24 GB VRAM (e.g., RTX 3090, RTX 4090, A5000, H100) |
| **OS**      | Linux (tested); other platforms unverified                          |

> ⚠️ **Note**: GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors.

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
conda create -n ar1 python=3.12 -y
conda activate ar1
uv sync --active
```

`uv sync --active` creates the project environment at `.venv` and installs the
locked project dependencies. Activate it before running the scripts:

```bash
source .venv/bin/activate
```

### 3. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:

- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate using the HuggingFace CLI:

```bash
pip install -U huggingface_hub
hf auth login
```

Get your access token at: https://huggingface.co/settings/tokens

> 💡 **Tip**: For more details on HuggingFace authentication, see the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo_r1/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to change
the `num_traj_samples=1` argument to a higher number (Line 60).

### Interactive notebook

We provide a notebook with similar inference code at `notebook/inference.ipynb`.

## This Clone: Local Changes

This repository is based on the NVIDIA Alpamayo 1 release and adds research
utilities for studying the relationship between Chain-of-Causation reasoning
and trajectory actions.

### Code changes

| File | Change |
| ---- | ------ |
| `src/alpamayo_r1/helper.py` | Supports injecting a forced reasoning trace into the VLM prompt. |
| `src/alpamayo_r1/models/alpamayo_r1.py` | Adds forced-reasoning trajectory sampling and returns the normalized unicycle controls used for intervention. |
| `src/alpamayo_r1/test_inference.py` | Keeps the original inference example and exposes trajectory/action information for inspection. |
| `src/alpamayo_r1/evaluate_reasoning_action.py` | Evaluates whether a predicted trajectory is consistent with a reasoning trace using an OpenAI-compatible language-model judge. It also reports timed waypoints and kinematic evidence. |
| `src/alpamayo_r1/evaluate_reasoning_intervention.py` | Runs paired reasoning interventions, saves full actions/waypoints, supports resume, and writes per-clip progress logs and a final summary. |

The intervention evaluator compares a clean action `u1` with a perturbed action
`u2` and applies:

```text
u_new = u1 + alpha * (u1 - u2)
```

For Alpamayo, each control contains acceleration and curvature over 64 time
steps. The evaluator supports `no_reasoning`, `noisy`, `cross_scene`, and
`opposite_action` modes. The current intervention benchmark is a sensitivity
test; it does not prove that a changed trajectory is safer or closer to the
ground truth.

### Setup used for the clone

The following setup was tested on Linux with an RTX 3090 and Python 3.12:

```bash
conda create -n ar1 python=3.12 -y
conda activate ar1
uv sync --active
```

If `uv sync --active` fails while building `flash-attn` because `nvcc` is not
available, use PyTorch SDPA instead:

```bash
sed -i '/"flash-attn>=2.8.3",/d' pyproject.toml
sed -i 's/attn_implementation: str = "flash_attention_2"/attn_implementation: str = "sdpa"/' \
  src/alpamayo_r1/models/base_model.py
uv sync --active
```

For the editable local package used by the evaluation scripts:

```bash
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

Authenticate before downloading gated model/data resources:

```bash
uv pip install --python .venv/bin/python -U huggingface_hub
hf auth login
```

### Basic inference

```bash
python src/alpamayo_r1/test_inference.py
```

The released model requires access to `nvidia/Alpamayo-R1-10B` and the
Physical AI AV dataset. The first run downloads several large model shards.

## Reasoning-Action Evaluation

### Original reasoning-action evaluation

Create a text file with one clip ID per line, then run:

```bash
python src/alpamayo_r1/evaluate_reasoning_action.py \
  --clip-ids-file test_clip_ids.txt \
  --output reasoning_action_eval.jsonl \
  --lm-model gpt-5.6-luna \
  --base-url https://api.xah.io/v1
```

The API key is requested interactively and is not stored in the output files.
The judge receives the reasoning and the predicted timed trajectory. This
evaluation does not use a ground-truth trajectory; it measures reasoning-action
alignment only.

### Intervention benchmark

For a 300-clip fixed-alpha baseline:

```bash
python src/alpamayo_r1/evaluate_reasoning_intervention.py \
  --clip-ids-file test_clip_ids_300.txt \
  --output reasoning_intervention_300_waypoints.jsonl \
  --modes no_reasoning noisy cross_scene opposite_action \
  --noisy-strategy lm_conflict \
  --noisy-model gpt-5.6-luna \
  --alphas 0 0.5 1.0 2.0 \
  --lm-model gpt-5.6-luna \
  --base-url https://api.xah.io/v1
```

The expected maximum is:

```text
300 clips × 4 modes × 4 alpha values = 4,800 records
```

`alpha=0` is the clean-action baseline. The evaluator writes each completed
record immediately to JSONL, so an interrupted run can be resumed by rerunning
the same command. Do not use `--overwrite` when resuming. Clean reasoning
traces are cached in:

```text
<output_stem>_reasonings.jsonl
```

LM-generated noisy traces are cached separately in:

```text
<output_stem>_noisy_reasonings.jsonl
```

The final summary is written only after all requested records finish. Each
record contains the clean/perturbed/guided control summaries, raw normalized
controls, 64 timed waypoints, guidance changes, saturation statistics, and the
LM judge result.

### Intervention modes

| Mode | Description |
| ---- | ----------- |
| `no_reasoning` | Runs the perturbed branch without a reasoning trace. |
| `noisy` | Uses a contradictory reasoning trace generated by the selected noise strategy. |
| `cross_scene` | Uses the reasoning trace from another clip; at least two clip IDs are required. |
| `opposite_action` | Appends an explicit contradictory action to the clean reasoning. |

For LM-generated noise, the same API key and compatible endpoint can be used
for both noise generation and judging. For a cheaper deterministic test, use:

```bash
--noisy-strategy heuristic_conflict
```

### Inspecting one clip

```bash
python - <<'PY'
import json

clip_id = "YOUR-CLIP-ID"
path = "reasoning_intervention_300_waypoints.jsonl"

with open(path, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        if record.get("clip_id") != clip_id:
            continue
        judge = record.get("lm_judgement_guided_action", {})
        print(record["mode"], record["alpha"])
        print("clean:", record.get("clean_reasoning"))
        print("perturbed:", record.get("perturbed_reasoning"))
        print("score:", judge.get("consistency_score"))
        print("label:", judge.get("label"))
        print("evidence:", judge.get("trajectory_evidence"))
PY
```

Use `lm_judgement_guided_action.consistency_score` for the per-record score.
The score is a language-model assessment, not a driving-safety guarantee.

## Relationship with the Paper

Alpamayo 1 implements the architecture described in our paper [*"Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088), including:

| Feature                                 | Paper Description                                                | This Release (v1.0)    |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Included            |
| **SFT fine-tuning (weights)**           | SFT trained model weights                                        | ✅ Included            |
| **SFT fine-tuning (code)**              | Supervised fine-tuning pipeline                                  | ✅ Included            |
| **RL post-training (weights)**          | RL post-trained model weights                                    | ❌ Not in this release |
| **RL post-training (code)**             | RL post-training pipeline via Cosmos-RL                          | ✅ Included            |
| **Route/navigation conditioning**       | Explicit navigation or route inputs                              | ❌ Not in this release |
| **Meta-actions/General VQA**            | High-level behavior and visual question answering                | ❌ Not in this release |

This release includes the core model, and the inference scripts. For SFT scripts, RL post-training pipeline, etc. please refer ro [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes).

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>Does the 10B model accept navigation/route inputs?</strong></summary>

While we have experimented with route conditioning capabilities, the released model does **not** include this feature. The current release takes multi-camera video and egomotion history as inputs, without explicit navigation or route inputs (e.g., waypoints, turn-by-turn navigation instructions).

</details>

<details>
<summary><strong>Does the model produce meta-actions or support general VQA?</strong></summary>

While we have experimented with meta-action and general VQA capabilities, the released model does **not** include these features. Alpamayo 1 is designed specifically for trajectory prediction with Chain-of-Causation reasoning, producing trajectory + reasoning trace outputs.

</details>

<details>
<summary><strong>Was the 10B model post-trained with Reinforcement Learning (RL)?</strong></summary>

No. The current 10B model release has **not** undergone RL post-training. While the paper describes RL stages for improving reasoning quality and action consistency, this release focuses on the supervised learning components. As mentioned above, we may release RL post-trained models in future releases.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, and H100. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

Yes. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

</details>

## Project Structure

```
alpamayo/
├── notebook/
│   └── inference.ipynb                  # Example notebook
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── common/
│       │   └── ...                      # logging utilities
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues:

```python
# Use PyTorch's scaled dot-product attention instead
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Ensure you have a GPU with at least 24 GB VRAM
2. Reduce `num_traj_samples` if generating multiple trajectories
3. Close other GPU-intensive applications

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: OpenMDW-1.1 - see the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.


## Disclaimer

Alpamayo 1 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
