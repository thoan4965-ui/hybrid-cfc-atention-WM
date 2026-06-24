# LeWM Project — Documentation Index (by Version)

## Docs by Version

### V0 — Bionic Hand 8-DOF (real robot)
| File | Nội dung |
|---|---|
| `docs/v0/index.md` | Tổng quan V0 |
| `docs/datasheets/sc09-servo-datasheet.md` | SC09 servo specs + calibration |
| `docs/datasheets/scservo-sdk-api.md` | Python API cho SCServo |
| `docs/datasheets/serial-protocol.md` | Serial COM13 timing, protocol |
| `le-wm-vo/robot/README.md` | Hardware setup guide |

### V1 — Hybrid CfC+Attention (abandoned)
| File | Nội dung |
|---|---|
| `docs/v1/index.md` | Tổng quan V1 + lý do abandon |

### V2.1 — Hybrid Mamba-2+Attention ✅ (main result)
| File | Nội dung |
|---|---|
| `docs/v2.1/index.md` | Tổng quan V2.1 + config |
| `docs/v2.1/module-specs.md` | Spec chi tiết predictor + embedder |
| `docs/v2.1/results-summary.md` | Bảng kết quả Push-T + TwoRoom |
| `docs/v2.1/setup-full.md` | Hướng dẫn cài đặt từ A→Z |
| `le-wm-v2.1/README.md` | Install + eval guide |
| `le-wm-v2.1/module.py` | Mamba2Predictor source |

### V2.5 — 4-DOF Robot Demo (lightweight deploy)
| File | Nội dung |
|---|---|
| `docs/v2.5/index.md` | Tổng quan V2.5 |
| `docs/v2.5/robot-specs.md` | Robot hardware specs (SG90, URDF, STL) |
| `docs/v2.5/deploy-pipeline.md` | Distill → Quant → ONNX → Edge |

### V2.9 — GA + Gradient + Hebbian + Dopamine ✅ (active)
| File | Nội dung |
|---|---|
| `docs/v2.9/index.md` | Tổng quan V2.9 — 4 cơ chế song song + dopamine điều phối |
| Source: `v2.6/v2_6/*.py` | 7 files, ~508 dòng, code trên GitHub |

### V3 — Multi-agent Future (conceptual)
| File | Nội dung |
|---|---|
| `docs/v3/index.md` | Tổng quan V3 roadmap |

### General (shared across versions)
| File | Nội dung | Priority |
|---|---|---|
| `docs/general/architecture-overview.md` | Kiến trúc tổng thể V0→V2.1→V3 | ⭐ cao |
| `docs/general/eval-protocol.md` | Protocol chuẩn cho eval (budget, seed, metric) | ⭐ cao |

## External references
| File/Path | Nội dung |
|---|---|
| `AGENTS.md` | Master protocol, 3-tier architecture, skills contracts |
| `plan/project_logbook.md` | Full logbook (rules, configs, results, changelog) |
| `plan/paper_links.md` | Danh sách paper đã đọc chia theo version |
| `link-paper/` | Paper summaries (51 files) |
| `skills/doc-reader/docs/` | Tool API references (SymPy, pyitlib, infomeasure, Golly, MuJoCo, NetLogo) |
| `plan/report/baocao_thuyetminh.md` | Báo cáo Sáng tạo trẻ hoàn chỉnh |
