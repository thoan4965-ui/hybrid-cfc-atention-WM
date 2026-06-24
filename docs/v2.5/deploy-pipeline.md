# V2.5 — Deploy Pipeline: PyTorch → Edge

## Step 1: Distill V2.1 → Student

```
Teacher (V2.1): 6×{Attn→Mamba-2}, 16.6M params
Student: 3×{Attn→Mamba-1}, 2-4M params
```

Distillation loss:
```
L = MSE(teacher_latent, student_latent) + MSE(teacher_pred, student_pred) + λ·KL(teacher_hidden, student_hidden)
```

Dataset: Push-T expert episodes (distilled + quantized does not need training from scratch).
Train on Colab T4: ~2h.

## Step 2: Quamba INT8 Quantization

Pipeline:
```
PyTorch model ──► torch.onnx.export() ──► ONNX graph
                                           │
                                           ▼
                                     Quamba quantizer
                                           │
                                           ▼
                                   INT8 ONNX model
                                           │
                                           ▼
                                     ONNX Runtime
```

Critical: Mamba-2 SSD kernel (Triton) is NOT ONNX-compatible.
Switch to Mamba-1 selective_scan (PyTorch-native) for ONNX export.
Quamba paper shows INT8 Mamba-1: <1% accuracy loss, 4x size reduction.

## Step 3: On-Device Inference

### Option A: Raspberry Pi 4/5
- CPU inference via ONNX Runtime
- Expected: 1-2 FPS (INT8, 4-core ARM)
- 4GB RAM sufficient
- No GPU needed

### Option B: NVIDIA Jetson Nano
- TensorRT INT8 inference
- Expected: 5-10 FPS
- 128 CUDA cores
- Better if available

### Option C: MCU (RP2350)
- Mamba-Lite-Micro (C codegen from ONNX)
- Expected: <1 FPS
- Proof-of-concept only

## Eval on Edge

Compare edge vs T4:
| Metric | T4 fp32 | RPi INT8 | Jetson INT8 |
|---|---|---|---|
| Push-T success | 94.7% | TBD | TBD |
| Speed (ms/ep) | 85s | TBD | TBD |
| Model size | 66 MB | ~16 MB | ~16 MB |
| RAM | 4 GB | 1-2 GB | 2-4 GB |

Goal: Push-T >80% on edge (acceptable drop from 94.7%).
