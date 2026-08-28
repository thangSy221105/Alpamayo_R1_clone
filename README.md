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
