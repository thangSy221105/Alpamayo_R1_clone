# Alpamayo-R1 Clone – Reasoning và Action Evaluation

Repository này là bản clone Alpamayo-R1, được bổ sung công cụ nghiên cứu mối
quan hệ giữa reasoning và trajectory action trong autonomous driving.

## Các thay đổi chính

- `src/alpamayo_r1/helper.py`: hỗ trợ chèn forced reasoning vào prompt VLM.
- `src/alpamayo_r1/models/alpamayo_r1.py`: sinh trajectory từ forced reasoning
  và trả về control chuẩn hóa.
- `src/alpamayo_r1/test_inference.py`: script kiểm tra inference trên GPU.
- `src/alpamayo_r1/evaluate_reasoning_action.py`: chấm độ bám giữa reasoning và
  trajectory bằng LM judge, kèm waypoint và bằng chứng kinematic.
- `src/alpamayo_r1/evaluate_reasoning_intervention.py`: chạy intervention,
  lưu action/waypoint, log từng clip, summary và resume.

Mỗi action của Alpamayo gồm 64 timestep với hai control:

```text
u_i = [acceleration_i, curvature_i]
```

Công thức intervention:

```text
u_new = u1 + alpha * (u1 - u2)
```

`u1` là action từ reasoning sạch, `u2` là action từ reasoning bị can thiệp.
Đây là phép đo độ nhạy, không đảm bảo sửa được trajectory score thấp.

## Setup trên Linux/GPU

Yêu cầu khuyến nghị: Linux, Python 3.12 và GPU NVIDIA có ít nhất 24 GB VRAM
(ví dụ RTX 3090). Trước tiên cần được cấp quyền truy cập model và dataset gated
trên Hugging Face.

```bash
conda create -n ar1 python=3.12 -y
conda activate ar1
uv sync --active
source .venv/bin/activate
uv pip install --python .venv/bin/python -U huggingface_hub openai
hf auth login
```

Nếu máy không có `nvcc` và `uv sync` lỗi khi build `flash-attn`, dùng SDPA:

```bash
sed -i '/"flash-attn>=2.8.3",/d' pyproject.toml
sed -i 's/attn_implementation: str = "flash_attention_2"/attn_implementation: str = "sdpa"/' \
  src/alpamayo_r1/models/base_model.py
uv sync --active
uv pip install --python .venv/bin/python -e .
```

## Inference cơ bản

```bash
python src/alpamayo_r1/test_inference.py
```

Lần chạy đầu sẽ tải model weights và dữ liệu mẫu. Không ghi API key vào source
code; evaluator sẽ yêu cầu key trực tiếp trong terminal.

## Đánh giá reasoning-action

Tạo file clip ID, mỗi dòng một ID:

```text
test_clip_ids.txt
```

```bash
python src/alpamayo_r1/evaluate_reasoning_action.py \
  --clip-ids-file test_clip_ids.txt \
  --output reasoning_action_eval.jsonl \
  --lm-model gpt-5.6-luna \
  --base-url https://api.xah.io/v1
```

Điểm này đo reasoning-action alignment, không đo độ đúng với ground-truth
trajectory.

## Intervention benchmark

Các mode:

| Mode | Ý nghĩa |
| --- | --- |
| `no_reasoning` | Nhánh bị can thiệp không nhận reasoning. |
| `noisy` | Dùng reasoning đối lập do LM hoặc heuristic tạo ra. |
| `cross_scene` | Dùng reasoning của clip khác; cần ít nhất 2 clip. |
| `opposite_action` | Chèn trực tiếp hành động đối lập. |

Chạy baseline alpha cố định trên 300 clip:

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

Số record tối đa:

```text
300 clip × 4 mode × 4 alpha = 4.800 record
```

Nếu muốn tạo nhiễu không cần gọi LM:

```bash
--noisy-strategy heuristic_conflict
```

## Resume

Chạy lại đúng lệnh cũ và không dùng `--overwrite`. Evaluator sẽ bỏ qua từng
cặp đã hoàn thành:

```text
clip_id + mode + alpha
```

Các cache quan trọng:

```text
<output_stem>_reasonings.jsonl
<output_stem>_noisy_reasonings.jsonl
```

Phải giữ file `_reasonings.jsonl` để tái sử dụng reasoning sạch. Chỉ xóa
`_noisy_reasonings.jsonl` khi muốn tạo lại reasoning nhiễu.

## Các trường output quan trọng

```text
clean_reasoning
perturbed_reasoning
clean_action_u1
perturbed_action_u2
guided_action_u_new_values
clean_waypoints
perturbed_waypoints
guided_waypoints
lm_judgement_guided_action
```

Điểm LM nằm ở:

```text
lm_judgement_guided_action.consistency_score
```

Xem score theo nhóm baseline:

```text
delta = score(alpha) - score(alpha=0)
baseline thấp: score(alpha=0) < 0.50
baseline cao:  score(alpha=0) >= 0.75
```

Nhóm baseline thấp dùng để báo cáo các trường hợp model đã sai từ đầu. Nhóm
baseline cao dùng để đo intervention có làm hỏng action tốt hay không.

## Giới hạn

- Intervention không đảm bảo sửa được trajectory score thấp.
- `cross_scene` không đảm bảo hai cảnh tương đồng.
- Flow matching có thể tạo kết quả không hoàn toàn deterministic.
- LM score là đánh giá alignment, không phải bảo đảm an toàn lái xe.

## Dataset và cách chọn clip

Các script sử dụng dataset **Physical AI Autonomous Vehicles** của NVIDIA:

- [Dataset trên Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Model Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [Repository gốc của NVIDIA](https://github.com/NVlabs/alpamayo)

Dataset là gated dataset, nên tài khoản Hugging Face phải được cấp quyền trước
khi chạy inference. File `clip_index.parquet` là index chứa metadata và
`clip_id` của các clip. Evaluator chỉ cần file text gồm các `clip_id`, mỗi dòng
một ID; dữ liệu camera và nhãn egomotion tương ứng sẽ được loader tải theo ID.

### Tạo danh sách test 300 clip

Ví dụ với file index đã tải về tại `~/clip_index.parquet`:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

index = pd.read_parquet(Path.home() / "clip_index.parquet")

# Kiểm tra tên cột trước khi lọc.
print(index.columns.tolist())

ids = (
    index["clip_id"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .head(300)
)

ids.to_csv("test_clip_ids_300.txt", index=False, header=False)
print("Saved", len(ids), "clip IDs")
PY
```

Kiểm tra số lượng và ID trùng:

```bash
wc -l test_clip_ids_300.txt
sort -u test_clip_ids_300.txt | wc -l
```

Kết quả mong muốn là `300` và `300`. Nếu muốn lấy mẫu ngẫu nhiên có thể tái
lập, thay `.head(300)` bằng:

```python
index["clip_id"].dropna().astype(str).drop_duplicates().sample(
    n=300, random_state=2026
)
```

Không nên lấy clip kế tiếp trong danh sách để suy ra hai cảnh gần nhau về ngữ
nghĩa. Trong mode `cross_scene`, clip kế tiếp chỉ được dùng như một nguồn
reasoning khác, không có bảo đảm cùng tuyến, cùng bối cảnh hay cùng hành động.

## LM evaluation: schema và prompt

LM judge nhận reasoning sạch cùng trajectory dự đoán. Dữ liệu gửi cho judge có
dạng rút gọn:

```json
{
  "reasoning": "Stop at the stop line because the light is red.",
  "coordinate_note": "Positive x is forward; positive y is left; 0.1s per waypoint.",
  "predicted_trajectory": {
    "start_xy_m": [0.0, 0.0],
    "end_xy_m": [7.3, 0.0],
    "path_length_m": 7.3
  },
  "trajectory_waypoints": [
    {"t_s": 0.0, "x_m": 0.0, "y_m": 0.0},
    {"t_s": 0.1, "x_m": 0.1, "y_m": 0.0}
  ],
  "trajectory_kinematics": {
    "segment_speed_min_mps": 0.0,
    "segment_speed_max_mps": 4.2,
    "segment_speed_final_mps": 0.01
  }
}
```

Prompt chính trong `evaluate_reasoning_action.py` yêu cầu LM:

```text
Evaluate whether the predicted driving action is consistent with the reasoning.
Judge action-reasoning alignment, not whether the reasoning is factually correct
about the scene. Do not judge whether the action is correct for the real scene
because no ground-truth trajectory or scene annotation is provided. Use all timed
waypoints and kinematics: for a stop, inspect low segment speeds and whether
forward position stops changing; for turn/lane actions, inspect the lateral path
using the supplied y-direction convention. Do not invent obstacle evidence.
The consistency_score is a continuous alignment score, not evaluator confidence.
Return exactly one JSON object with these fields:
consistency_score, label, intended_action, trajectory_evidence, uncertainty.
```

Schema bắt buộc của LM judge:

```json
{
  "consistency_score": 0.0,
  "label": "consistent | partially_consistent | inconsistent | uncertain",
  "intended_action": "string",
  "trajectory_evidence": "string",
  "uncertainty": "string"
}
```

Các khoảng điểm chỉ là hướng dẫn mềm:

```text
consistent:           khoảng 0.75–1.00
partially_consistent: khoảng 0.30–0.74
inconsistent:         khoảng 0.00–0.29
uncertain:            khoảng 0.25–0.70
```

Điểm được đọc từ:

```text
lm_judgement_guided_action.consistency_score
```

## Cấu hình noise và alpha

### Các kiểu noise

| Cấu hình | Cách tạo nhiễu |
| --- | --- |
| `irrelevant` | Thêm một ghi chú không liên quan; không tạo action đối lập rõ ràng. |
| `heuristic_conflict` | Đảo hành động bằng luật cố định, không gọi LM. |
| `lm_conflict` | Gọi LM để tạo action đối lập ngắn và lưu cache. |

Prompt tạo nhiễu ở `lm_conflict` yêu cầu:

```text
Create the opposite driving action for a robustness test. Map LEFT to RIGHT and
RIGHT to LEFT. Map ACCELERATE to DECELERATE and DECELERATE to ACCELERATE. Use a
concrete short action, not a refusal. Both fields must describe the same opposite
action. Return exactly two JSON fields: primary_action and noisy_reasoning.
Keep every value under 12 words.
```

Schema tạo nhiễu:

```json
{
  "primary_action": "Nudge right",
  "noisy_reasoning": "LEFT maps to RIGHT; opposite steering direction."
}
```

### Alpha

Alpha được truyền qua command line, có thể dùng nhiều giá trị trong một lần
chạy:

```bash
--alphas 0 0.5 1.0 2.0
```

Diễn giải:

- `alpha=0`: baseline, `u_new = u1`;
- alpha nhỏ: can thiệp nhẹ;
- alpha lớn: khuếch đại mạnh khác biệt giữa `u1` và `u2`, có thể làm hỏng
  trajectory;
- alpha lớn không bảo đảm sửa được clip có baseline score thấp.

## Lệnh chạy đầy đủ

### Kiểm tra syntax trước khi chạy

```bash
python -m py_compile \
  src/alpamayo_r1/helper.py \
  src/alpamayo_r1/models/alpamayo_r1.py \
  src/alpamayo_r1/evaluate_reasoning_action.py \
  src/alpamayo_r1/evaluate_reasoning_intervention.py
```

### Chạy baseline fixed-alpha 300 clip

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

### Chạy chỉ số liệu số học, không gọi LM judge

```bash
python src/alpamayo_r1/evaluate_reasoning_intervention.py \
  --clip-ids-file test_clip_ids_300.txt \
  --output intervention_numeric_only.jsonl \
  --modes no_reasoning noisy cross_scene opposite_action \
  --noisy-strategy heuristic_conflict \
  --alphas 0 0.5 1.0 2.0 \
  --skip-lm
```

### Các tham số thường dùng

```text
--clip-ids-file       file ID đầu vào
--output              file JSONL kết quả
--t0-us               thời điểm bắt đầu clip, mặc định 5100000
--alphas              một hoặc nhiều alpha không âm
--modes               các mode intervention
--noisy-strategy      irrelevant, heuristic_conflict hoặc lm_conflict
--noisy-model         model tạo nhiễu, mặc định dùng --lm-model
--lm-model            model judge
--base-url            OpenAI-compatible API endpoint
--positive-y-direction left hoặc right
--skip-lm             chỉ chạy metrics số học
--overwrite           xóa output/cache liên quan và chạy lại từ đầu
```

Khi resume, chạy lại đúng command nhưng không thêm `--overwrite`.
