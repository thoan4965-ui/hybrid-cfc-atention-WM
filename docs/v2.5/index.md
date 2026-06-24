# V2.5 — 4-DOF Robot Demo (Lightweight Model Deploy)

## Goal
Build 4-DOF robot arm (SG90/MG90S servos, 3D-printed frame) running lightweight world model for pick-and-place demo. Target: real-time inference on edge (Raspberry Pi or Jetson Nano).

## Strategy
1. **Distill** V2.1 full model → lightweight student (TinyViT encoder + mini Mamba predictor)
2. **Quantize** with Quamba ONNX INT8 → 4x size reduction
3. **Deploy** on edge with ONNX Runtime or TensorRT
4. **Demo** real-time pick-and-place objects >2cm

## Architecture

```
V2.1 Teacher (16.6M params) ──distill──► V2.5 Student (2-4M params)
                                               │
                                               ▼
                                        Quamba INT8 quant
                                               │
                                               ▼
                                     ONNX Runtime / TensorRT
                                               │
                                               ▼
                                     Raspberry Pi / Jetson Nano
                                               │
                                               ▼
                                     Robot 4-DOF (SG90 servos)
```

## Components

| Component | Choice | Size | Notes |
|---|---|---|---|
| Encoder | TinyViT-5M or Vim-S | 5-6M | Distilled from V2.1 ViT-tiny |
| Predictor | mini Mamba-2 (3 blocks) | 1-2M | Half depth of V2.1 |
| Quantization | Quamba INT8 | 4x smaller | ONNX graph |
| Inference | ONNX Runtime | — | CPU + optional TensorRT |
| Robot | 4-DOF arm SG90 | $15 | 3D-printed frame |

## Papers

See `plan/paper_links.md` section V2.5:
- CompACT (visual obs compression)
- MambaVision (lightweight encoder)
- Quamba (Mamba INT8 quantization)
- SLiM (combined compression)
- Vim/VMamba (SSM vision backbone)
- TinyViT (efficient vision transformer)
- Knowledge Distillation (model distillation)

## Dependencies

| Tool | Use | Docs |
|---|---|---|
| MuJoCo | Sim robot physics | `skills/doc-reader/docs/mujoco.md` |
| Quamba | Mamba ONNX quantization | `link-paper/33` |
| ONNX Runtime | Edge inference | `link-paper/39` |
| FABRI_CREATOR | STL for 3D print | `plan/links.md` |
