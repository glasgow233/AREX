"""Self-samples of SANA-0.6B (512px, classifier-free guidance w = 4.5): the raw material of the AREX-C moments.

SANA's training set is not released, so the terminal law of the guided model is sampled with the model's own
pipeline (factory scheduler and grid, --steps sampling steps, bf16) on 2000 synthetic template prompts,
--n / 2000 samples per prompt.  The ODE endpoint latents are stored in shards of 1000 in --out (z_0000.pt, ...),
together with the prompt list (prompts.json).  A short check of the pipeline conventions is printed first.

The paper uses --n 50000 (25 samples per prompt) and --steps 20.
"""
import os, sys, time, json, argparse, traceback
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50000, help="number of latents (2000 prompts, n / 2000 per prompt)")
ap.add_argument("--bs", type=int, default=25)
ap.add_argument("--steps", type=int, default=20, help="sampling steps of the factory scheduler")
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--out", default="runs/sana512_latents")
A = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A.out = A.out if os.path.isabs(A.out) else os.path.join(ROOT, A.out)      # relative to the repo root
t0 = time.time(); el = lambda: f"[{(time.time()-t0)/60:6.1f} min]"
MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
DEV = "cuda"

# ---------------------------------------------------------------- 1. load
from diffusers import SanaPipeline
print(f"{el()} loading {MODEL} ...", flush=True)
pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(DEV)
pipe.set_progress_bar_config(disable=True)
tr, vae = pipe.transformer, pipe.vae
print(f"{el()} transformer {sum(p.numel() for p in tr.parameters())/1e9:.2f} B  "
      f"vae {sum(p.numel() for p in vae.parameters())/1e9:.2f} B", flush=True)

# ---------------------------------------------------------------- 2. conventions: latent shape, grid, NFE per step
print(f"\n{el()} === pipeline conventions ===", flush=True)
calls = {"n": 0}
_orig = tr.forward
def counted(*a, **k):
    calls["n"] += 1
    return _orig(*a, **k)
tr.forward = counted
grab = {}
def cb(pipe_, step, timestep, kw):
    if step == 0:
        grab["latent_shape"] = tuple(kw["latents"].shape)
        grab["sigmas"] = [float(s) for s in pipe_.scheduler.sigmas[:6]]
        grab["sigma_last"] = float(pipe_.scheduler.sigmas[-1])
    return kw
calls["n"] = 0
img = pipe("a photo of a red apple on a wooden table", num_inference_steps=A.steps,
           guidance_scale=A.cfg, generator=torch.Generator(DEV).manual_seed(0),
           callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
d = int(np.prod(grab["latent_shape"][1:]))
print(f"  latent shape      {grab['latent_shape']}   ->  d = {d}")
print(f"  scheduler         {type(pipe.scheduler).__name__}  (flow_shift {pipe.scheduler.config.flow_shift})")
print(f"  sigmas[:6]        {[round(s,4) for s in grab['sigmas']]}   last {grab['sigma_last']:.4f}")
print(f"  t grid = 1-sigma  {[round(1-s,4) for s in grab['sigmas']]}")
print(f"  vae scaling       {vae.config.scaling_factor}")
print(f"  NFE (steps={A.steps}, cfg={A.cfg})  {calls['n']}  -> {calls['n']/A.steps:.1f} per step (1 = batched CFG)", flush=True)
tr.forward = _orig

# ---------------------------------------------------------------- 3. latents
os.makedirs(A.out, exist_ok=True)
PROMPTS = os.path.join(A.out, "prompts.json")
if os.path.exists(PROMPTS):
    prompts = json.load(open(PROMPTS))
else:
    # 2000 synthetic template prompts (subject, style, detail), disjoint from MJHQ-30K
    subj = ["a cat","a mountain lake","an old library","a robot","a bowl of fruit","a city street",
            "a forest path","a sailing boat","a violin","a desert dune","a glass of water","a red fox",
            "a cathedral","a bicycle","a plate of sushi","a snowy village","a lighthouse","a hot air balloon",
            "a bookstore","a waterfall"]
    style = ["photorealistic","oil painting","watercolor","cinematic lighting","macro photo","pencil sketch",
             "3d render","vintage photograph","studio portrait","golden hour"]
    extra = ["highly detailed","soft focus","dramatic shadows","vivid colors","minimalist","misty",
             "wide angle","close up","warm tones","cool tones"]
    rng = np.random.default_rng(0)
    prompts = [f"{subj[rng.integers(len(subj))]}, {style[rng.integers(len(style))]}, {extra[rng.integers(len(extra))]}"
               for _ in range(2000)]
    json.dump(prompts, open(PROMPTS, "w"))
if A.n % len(prompts):
    print(f"  WARNING: n={A.n} is not a multiple of {len(prompts)} prompts; sana_moments.py needs a multiple "
          f"(it groups latents by prompt)", flush=True)
print(f"\n{el()} === sampling {A.n} latents, bs={A.bs} ===", flush=True)

@torch.no_grad()
def run_batch(ps, seed):
    """Run the pipeline and grab the ODE endpoint latents before the VAE decode."""
    out = {}
    def cb2(pipe_, step, timestep, kw):
        if step == A.steps - 1:
            out["z"] = kw["latents"].detach().float().cpu()
        return kw
    pipe(ps, num_inference_steps=A.steps, guidance_scale=A.cfg,
         generator=torch.Generator(DEV).manual_seed(seed),
         output_type="latent", callback_on_step_end=cb2,
         callback_on_step_end_tensor_inputs=["latents"])
    return out["z"]

shards = sorted(p for p in os.listdir(A.out) if p.startswith("z_"))
start = sum(torch.load(os.path.join(A.out, p), mmap=True).shape[0] for p in shards)
print(f"  {len(shards)} shards present ({start} latents), starting at {start}", flush=True)
buf = []
try:
    for i0 in range(start, A.n, A.bs):
        ps = [prompts[(i0 + j) % len(prompts)] for j in range(min(A.bs, A.n - i0))]   # prompt of latent i = i % 2000
        buf.append(run_batch(ps, 900000 + i0))
        if sum(b.shape[0] for b in buf) >= 1000:
            Z = torch.cat(buf); buf = []
            k = len([p for p in os.listdir(A.out) if p.startswith("z_")])
            torch.save(Z, os.path.join(A.out, f"z_{k:04d}.pt"))
            print(f"  {el()} saved z_{k:04d}.pt  {tuple(Z.shape)}  ~{(k + 1) * 1000} so far", flush=True)
except Exception:
    traceback.print_exc()
    print(f"{el()} interrupted; the shards written so far remain usable", flush=True)
if buf:                                                     # flush the last, shorter shard
    Z = torch.cat(buf); k = len([p for p in os.listdir(A.out) if p.startswith("z_")])
    torch.save(Z, os.path.join(A.out, f"z_{k:04d}.pt"))
    print(f"  {el()} saved z_{k:04d}.pt  {tuple(Z.shape)} (final shard)", flush=True)
print(f"\n{el()} done")
