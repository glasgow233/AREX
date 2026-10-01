"""AREX-C moments for SANA-0.6B from the self-sampled latents of sana_selfsample.py.

A) Within-prompt covariance S_bar = E_c[Cov(x | c)] and pooled mean m1.  With N latents over P = 2000 prompts
   (prompt of latent i = i % P) there are N / P samples per prompt; S_bar is the unbiased pooled estimate,
   symmetrised and eigendecomposed.  Output: <out>/sbar_N<N>.pt with keys m1, lam, U.
B) Calibration coefficient alpha of the affine center mu(c) = m1 + alpha (x1h(t0) - m1): least squares of
   mu_p - m1 (empirical per-prompt mean) on x1h(t0) - m1 (endpoint prediction of the guided network at the
   first node of the 20-step factory grid), over prompts 512:1024 of the synthetic list.  Output: <out>/alpha.json.

Requires the Hugging Face checkpoint Efficient-Large-Model/Sana_600M_512px_diffusers.
"""
import os, sys, json, glob, time, inspect, argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/sana512_latents")
ap.add_argument("--prompts", type=int, default=2000, help="number of distinct prompts P")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--alpha_prompts", default="512:1024", help="prompt slice used to fit alpha")
A = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A.out = A.out if os.path.isabs(A.out) else os.path.join(ROOT, A.out)      # relative to the repo root
t0 = time.time(); el = lambda: f"[{(time.time()-t0)/60:5.1f}m]"
OUT, P, DEV = A.out, A.prompts, "cuda"

# ---------------------------------------------------------------- A) S_bar and m1
fs = sorted(glob.glob(f"{OUT}/z_*.pt"))
Z = torch.cat([torch.load(f) for f in fs]).flatten(1).double()
N, d = Z.shape
reps = N // P
print(f"{el()} Z {tuple(Z.shape)}  {P} prompts x {reps} samples", flush=True)
assert reps * P == N, f"{N} is not a multiple of {P}: the prompt grouping would be misaligned"

Zg = Z.reshape(reps, P, d)          # latent [k, j] belongs to prompt (k*P + j) % P = j
mu_p = Zg.mean(0)                   # (P, d) per-prompt mean
m1 = Z.mean(0)

Zc = (Zg - mu_p).reshape(-1, d)
S = (Zc.T @ Zc) / (P * (reps - 1))              # unbiased within-prompt covariance
Mc = mu_p - m1
Sb = (Mc.T @ Mc) / (P - 1)                      # between-prompt covariance
print(f"{el()} tr S_bar (within) = {float(S.trace()):.1f}   tr Cov(mu) (between) = {float(Sb.trace()):.1f}   "
      f"sum = {float(S.trace()+Sb.trace()):.1f}", flush=True)
Zt = Z - m1
print(f"        tr Sigma_1 (pooled) = {float((Zt.T@Zt).trace()/(N-1)):.1f}  (should agree with the sum)")
print(f"        between-prompt share = {float(Sb.trace()/(S.trace()+Sb.trace())):.1%}", flush=True)

S = 0.5 * (S + S.T)
lam, Q = torch.linalg.eigh(S)
lam = lam.clamp_min(1e-8)
lp = lam.numpy()[::-1].copy()
print(f"\n{el()} S_bar spectrum: max={lp[0]:.3f}  median={np.median(lp):.4f}  min={lp[-1]:.2e}   "
      f"propagator gain sqrt(1+lam_max) = {np.sqrt(1+lp[0]):.2f}")
cum = np.cumsum(lp)/lp.sum()
for f_ in (0.5, 0.9, 0.99):
    print(f"        {f_:.0%} of the variance in {int(np.searchsorted(cum,f_))+1} / {d} directions")
torch.save({"m1": m1.float().cpu(), "lam": lam.float().cpu(), "U": Q.float().cpu(),
            "N": N, "d": d, "kind": "within_prompt"}, f"{OUT}/sbar_N{N}.pt")
print(f"{el()} saved {OUT}/sbar_N{N}.pt", flush=True)

# ---------------------------------------------------------------- B) alpha
print(f"\n{el()} === alpha calibration on prompts {A.alpha_prompts} ===", flush=True)
from diffusers import SanaPipeline
pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px_diffusers",
                                    torch_dtype=torch.bfloat16).to(DEV)
pipe.set_progress_bar_config(disable=True)
tr, sch = pipe.transformer, pipe.scheduler
tr.to(torch.float32)                      # fp32 for the calibration (bf16 floor ~1.6e-2)
CHI = inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default
prompts = json.load(open(f"{OUT}/prompts.json"))

def field(ps):
    with torch.no_grad():
        pe, pm, ne, nm = pipe.encode_prompt(ps, do_classifier_free_guidance=True, device=DEV,
                                            clean_caption=False, max_sequence_length=300,
                                            complex_human_instruction=CHI)
    emb = torch.cat([ne, pe]).to(tr.dtype); msk = torch.cat([nm, pm])
    @torch.no_grad()
    def v(t, xf):
        x = xf.reshape(-1, 32, 16, 16)
        o = tr(torch.cat([x, x]).to(tr.dtype), encoder_hidden_states=emb,
               encoder_attention_mask=msk,
               timestep=torch.full((2*x.shape[0],), (1.0-float(t))*1000.0, device=DEV),
               return_dict=False)[0].float()
        u, c = o.chunk(2)
        return -(u + A.cfg*(c-u)).reshape(x.shape[0], -1)
    return v

def grid(n):
    sch.set_timesteps(n, device=DEV)
    return (1.0 - sch.sigmas.float().cpu().numpy())

lo, hi = [int(v) for v in A.alpha_prompts.split(":")]
m1d = m1.float().to(DEV)
num = den = 0.0
for i0 in range(lo, hi, A.bs):
    ps = prompts[i0:i0+A.bs]; f = field(ps)
    x0 = torch.randn(len(ps), d, device=DEV, generator=torch.Generator(DEV).manual_seed(77+i0))
    tt = float(grid(20)[0])                        # first node of the 20-step factory grid
    xt = (1-tt)*x0
    x1h = xt + (1-tt)*f(tt, xt)                    # endpoint prediction of the guided network
    a = x1h - m1d                                  # predicted displacement
    b = mu_p[i0:i0+A.bs].float().to(DEV) - m1d     # empirical displacement
    num += float((a*b).sum()); den += float((a*a).sum())
alpha = num/den
print(f"  alpha = <x1h-m1, mu_p-m1> / ||x1h-m1||^2 = {alpha:.4f}", flush=True)
json.dump({"alpha": alpha, "prompts": A.alpha_prompts, "t0": tt, "N": N, "cfg": A.cfg},
          open(f"{OUT}/alpha.json", "w"), indent=1)
print(f"{el()} saved {OUT}/alpha.json")
