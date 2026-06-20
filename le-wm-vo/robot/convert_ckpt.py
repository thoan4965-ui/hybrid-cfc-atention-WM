"""Convert fine-tune checkpoint to flat format (compatible with test_encoder, robot_planner)."""
import torch, sys
ckpt = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else ckpt

sd = torch.load(ckpt, map_location='cpu')
if 'state_dict' in sd:
    inner = sd['state_dict']
else:
    inner = sd

flat = {}
for comp in ['encoder.', 'predictor.', 'action_encoder.']:
    if comp.rstrip('.') in inner:
        for k, v in inner[comp.rstrip('.')].items():
            flat[f'{comp}{k}'] = v

flat_sd = {'state_dict': {f'model.{k}': v for k, v in flat.items()}}
torch.save(flat_sd, out)
print(f"Converted {ckpt} -> {out}")
print(f"  Keys: {len(flat_sd['state_dict'])}")
for k in list(flat_sd['state_dict'].keys())[:5]:
    print(f"  {k}: {flat_sd['state_dict'][k].shape}")
