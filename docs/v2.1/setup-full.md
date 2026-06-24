# Setup Guide — V2.1

## 1. Clone repo

```bash
git clone https://github.com/thoan4965-ui/hybrid-mamba-atention-WM
cd hybrid-mamba-atention-WM
```

## 2. Python env

```bash
# Python 3.12 recommended (3.14 có thể incompatible với một số packages)
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Hoặc: .venv\Scripts\Activate  # Windows
```

## 3. Install dependencies

```bash
# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# LeWM framework
pip install stable-pretraining stable-worldmodel huggingface_hub hydra-core einops imageio

# Mamba-2
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 4. Download datasets

```bash
# Push-T
huggingface-cli download quentinll/lewm-pusht --local-dir data/pusht

# TwoRoom
huggingface-cli download quentinll/lewm-tworooms --local-dir data/tworoom

# Giải nén
zstd -d data/pusht/pusht_expert_train.h5.zst
zstd -d data/tworoom/tworoom.tar.zst && tar xf data/tworoom/tworoom.tar
```

## 5. Set environment

```bash
export STABLEWM_HOME=/content/data  # Colab
# Hoặc set trong Python:
import os; os.environ['STABLEWM_HOME'] = '/path/to/data'
```

## 6. Eval

```bash
cd le-wm-v2.1
python eval.py task=pusht checkpoint.path=checkpoints/hybrid_mamba_pusht.pt
```

Chi tiết: `le-wm-v2.1/README.md`

## 7. Train (nếu cần)

```bash
cd le-wm-v2.1
python train.py task=pusht model=predictor/mamba2
```
