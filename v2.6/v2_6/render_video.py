"""Render video with food overlay — flexible path, checkpoint or result."""
import os, time, sys
os.environ['MUJOCO_GL'] = 'glx'; os.environ['DISPLAY'] = ':99'
from v2_6.main import env, genome_to_policy, policy_forward
import jax, jax.numpy as jnp, numpy as np, mediapy as media
from jax import lax, jit
import mujoco, mujoco.mjx as mjx
from PIL import Image, ImageDraw

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else 'v2_6/results/v29_0624_1500.npz'
N_FRAMES = 200; W = 320; H = 240

data = np.load(DATA_PATH, allow_pickle=True)
if 'best_nodes' in data:
    n, c = data['best_nodes'], data['best_conns']
else:
    n, c = data['nodes'][0], data['conns'][0]
pol = genome_to_policy(jnp.array(n), jnp.array(c))

@jit
def rollout(pol, key):
    def step(s, _):
        a, _ = policy_forward(pol, s.obs); s2 = env.step(s, a)
        return s2, (s2.pipeline_state, s2.info['food_pos'], s2.info['food_cnt'])
    _, (states, fps, fcs) = lax.scan(step, env.reset(key), jnp.arange(N_FRAMES))
    return states, fps, fcs

print("Rollout GPU...", flush=True)
t0 = time.time()
states, fps, fcs = rollout(pol, jax.random.PRNGKey(0))
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

renderer = mujoco.Renderer(env.sys.mj_model, width=W, height=H)
frames = []
print("Render CPU + food overlay...", flush=True)
t0 = time.time()

for i in range(N_FRAMES):
    mj_data = mjx.get_data(env.sys.mj_model, states[i])
    renderer.update_scene(mj_data)
    frame = renderer.render().copy()

    fp = np.array(fps[i]); fc = np.array(fcs[i])
    px = ((fp[:, 0] + 10) / 20 * W).astype(int)
    py = (H - (fp[:, 1] + 10) / 20 * H).astype(int)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    for j in range(len(fp)):
        color = 'red' if fc[j] > 0 else 'gray'
        draw.ellipse([px[j]-4, py[j]-4, px[j]+4, py[j]+4], fill=color)
    frames.append(np.array(pil))

    if (i+1) % 50 == 0:
        dt = time.time() - t0
        print(f"  {i+1}/{N_FRAMES}  {dt:.0f}s", flush=True)

renderer.close()
video = jnp.stack(frames)
print(f"  Done: {time.time()-t0:.0f}s, shape={video.shape}", flush=True)
media.show_video(video, fps=30, height=400)
