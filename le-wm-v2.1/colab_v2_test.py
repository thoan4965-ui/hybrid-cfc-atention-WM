# %% [markdown]
# # V2 Test — Mamba-3+Attention Hybrid Predictor
# Run this on Colab T4 (free) to verify code before RTX 5080.

# %% [markdown]
# ## 1. Install dependencies

# %%
!pip install -q git+https://github.com/state-spaces/mamba.git
!pip install -q stable-pretraining stable-worldmodel huggingface_hub hydra-core einops

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 2. Load module.py from le-wm-v2

# %%
import sys, os, json, shutil
from pathlib import Path
from einops import rearrange

# Clone le-wm-v2 if not already cloned
if not Path("le-wm-v2").exists():
    !git clone https://github.com/lucas-maes/le-wm.git le-wm-v2
sys.path.insert(0, "le-wm-v2")

# Import critical classes
from module import (
    Attention, ConditionalBlock, ARPredictor,
    Mamba3ConditionalBlock, Mamba3Predictor,
    Embedder, MLP,
)

print("Mamba3Predictor imported OK")

# %% [markdown]
# ## 3. Test Mamba-3 forward pass

# %%
B, T, D = 4, 4, 192

# Create Mamba3Predictor
predictor = Mamba3Predictor(
    num_frames=T,
    depth=6,
    heads=6,
    dim_head=64,
    input_dim=D,
    hidden_dim=D,
    d_state=128,
    headdim=64,
    dropout=0.1,
    emb_dropout=0.0,
).to(device)

# Test forward
x = torch.randn(B, T, D, device=device)
c = torch.randn(B, T, D, device=device)

with torch.no_grad():
    out = predictor(x, c)

print(f"Input:  {x.shape}")
print(f"Cond:   {c.shape}")
print(f"Output: {out.shape}")
print(f"Params: {sum(p.numel() for p in predictor.parameters()):,}")

assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
print("✅ Forward pass OK")

# %% [markdown]
# ## 4. Param check — Attention vs Mamba3 ratio

# %%
def count_params(mod):
    return sum(p.numel() for p in mod.parameters())

# Create a single block for comparison
block_ar = ConditionalBlock(dim=D, heads=6, dim_head=64, mlp_dim=2048)
block_v2 = Mamba3ConditionalBlock(dim=D, heads=6, dim_head=64, d_state=128, headdim=64)

print("=== Parameter ratio check ===")
print(f"ConditionalBlock (AR):  {count_params(block_ar):,}")
print(f"Mamba3ConditionalBlock: {count_params(block_v2):,}")
ratio = count_params(block_ar) / max(count_params(block_v2), 1)
print(f"Ratio (AR/V2): {ratio:.2f}:1  (target ~1:1)")

attn_only = count_params(block_ar.attn)
mamba3_only = count_params(block_v2.mamba3)
print(f"  Attention only: {attn_only:,}")
print(f"  Mamba3 only:    {mamba3_only:,}")
print(f"  Ratio A:M = {attn_only/mamba3_only:.2f}:1")

# %% [markdown]
# ## 5. Test mini training loop (32 batches)

# %%
# Create dummy dataset
torch.manual_seed(42)
batch_size = 32
T = 4
emb_dim = 192
act_dim = 8

dummy_emb = torch.randn(1000, T, emb_dim, device=device)
dummy_act = torch.randn(1000, T, act_dim, device=device)
dataset = torch.utils.data.TensorDataset(dummy_emb, dummy_act)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate full JEPA model
from jepa import JEPA

model = JEPA(
    encoder=lambda x: type('obj', (object,), {'last_hidden_state': x})(),  # dummy encoder
    predictor=Mamba3Predictor(
        num_frames=T, depth=6, heads=6, dim_head=64,
        input_dim=emb_dim, hidden_dim=emb_dim,
        d_state=128, headdim=64, dropout=0.1
    ).to(device),
    action_encoder=Embedder(input_dim=act_dim, emb_dim=emb_dim).to(device),
    projector=MLP(input_dim=emb_dim, hidden_dim=2048, output_dim=emb_dim, norm_fn=torch.nn.BatchNorm1d).to(device),
    pred_proj=MLP(input_dim=emb_dim, hidden_dim=2048, output_dim=emb_dim, norm_fn=torch.nn.BatchNorm1d).to(device),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"\nTotal model params: {sum(p.numel() for p in model.parameters()):,}")

# Train 32 batches
model.train()
losses = []
for i, (emb, act) in enumerate(loader):
    if i >= 32:
        break
    optimizer.zero_grad()
    
    # Create info dict as JEPA expects
    info = {"pixels": torch.randn(batch_size, T, 3, 224, 224, device=device)}
    info["action"] = act
    
    # Forward through JEPA
    with torch.no_grad():
        # Mock encoder (just returns emb)
        info["emb"] = emb
        info["act_emb"] = model.action_encoder(act)
    
    # Predict
    pred = model.predict(info["emb"], info["act_emb"])
    
    # Simple MSE loss
    loss = F.mse_loss(pred, emb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    if (i+1) % 10 == 0:
        print(f"  Step {i+1:2d}, loss = {loss.item():.6f}")

print(f"\n✅ Training loop OK. Final loss: {losses[-1]:.6f}")
print(f"Loss went from {losses[0]:.6f} → {losses[-1]:.6f} (down={losses[0]/losses[-1]:.2f}x)")

# %% [markdown]
# ## 6. Summary
# print("""
# === V2 Test Results ===
# 1. Mamba3Predictor forward:         ✅
# 2. Params ratio (A:M):              {:.2f}:1
# 3. Training loop:                    ✅
# 4. Ready for RTX 5080:              ✅
# """.format(attn_only/mamba3_only))
