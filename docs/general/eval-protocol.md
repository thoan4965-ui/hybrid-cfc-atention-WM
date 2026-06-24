# Eval Protocol — Standardized

Áp dụng cho mọi eval trong project. Thống nhất để fair comparison.

## Core parameters (mọi task)

| Param | Value | Ghi chú |
|---|---|---|
| **budget** | 50 | Số CEM iterations |
| **goal_offset** | 25 | Start planning từ step 25 |
| **H** | 5 | Horizon CEM |
| **K** | 5 | Number CEM candidates |
| **num_eval** | 50 | Số episode eval |
| **seed** | 3072-3074 | 3 seeds (3072, 3073, 3074) cho cả train + eval |
| **deterministic** | True | model.float(), torch.no_grad() |

## Tasks

| Task | Steps | Action dim | Goal |
|---|---|---|---|
| **TwoRoom** | 50 | 2 | Reach green area |
| **Push-T** | 100 | 2 | Push T to target |
| **Reacher** | 50 | 2 | Reach random target |
| **Cube** | 100 | 4 | Push cube to target |

## Metric

- **Success rate** = N success / N eval
- 95% **Wilson CI** cho mỗi kết quả
- Báo cáo: `X% (N/K, 95% Wilson CI: [low, high])`
- Luôn báo cáo **mean ± std** qua 3 seeds

## Pre-eval checklist

1. `zstd` installed
2. Dataset extracted + `STABLEWM_HOME` set
3. `sed` patch eval.py (nếu needed)
4. `weights_only=False` cho torch.load
5. `model.float()` sau load
6. CEM time: ghi riêng **first ep compile** vs **post-compile avg**

## Fair comparison rules

- So sánh phải cùng: T, budget, goal_offset, seed, task
- Cùng hardware (ghi rõ GPU type: T4, A100, RTX 5090)
- Báo cáo cả speed (CEM time) + accuracy
- Speed chỉ so sánh post-compile, ko tính first ep compile time
