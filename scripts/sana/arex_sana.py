"""AREX-C on SANA-0.6B (512px, MJHQ-30K, classifier-free guidance w = 4.5), following the SANA / STORK protocol:
30k prompts, seed 0, FID computed with SANA's metric toolkit against the MJHQ-30K reference statistics.

The sampler runs on SANA's factory time grid (DPMSolverMultistepScheduler, flow_shift = 3).  The within-prompt
covariance S_bar and the pooled mean m1 come from --sbar_file and the calibration coefficient alpha from
--alpha_file (both written by sana_moments.py).  The first network evaluation is at the first grid node and gives
the affine center mu(c) = m1 + alpha (x1h - m1) per sample; later nodes sit at u* = s + rho (t - s).  --scheme
selects the residual rule (frozen = one node, ab2, ab3); --phi adds the residual magnitude profile, measured once
along guided Euler-50 trajectories of the synthetic calibration prompts (never the MJHQ prompts).

Images are generated, scored and deleted per configuration (--keep keeps them).  Results are merged into
<out>/fid_n<n>.json under the key <tag>_nfe<k>, e.g. cfsc-ab2-rho0.35-phi_nfe4.

Requires: data/test/PG-eval-data/MJHQ-30K (mjhq_get.py), ext/Sana (the FID toolkit), the Hugging Face checkpoint
Efficient-Large-Model/Sana_600M_512px_diffusers, and the outputs of sana_selfsample.py / sana_moments.py.
"""
import os, sys, json, time, argparse, inspect, subprocess, numpy as np, torch
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from diffusers import SanaPipeline
from arex.surrogate import Surrogate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30000, help="number of MJHQ-30K prompts (the first n of meta_data.json)")
ap.add_argument("--bs", type=int, default=25)
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nfes", default="4,5,6,7,8,9,10")
ap.add_argument("--rho", type=float, default=0.5, help="node position inside the step (0.35 in the paper)")
ap.add_argument("--scheme", default="frozen", choices=["frozen", "ab2", "ab3"],
                help="residual rule: frozen = one node (W0 only); ab2 = + first Newton difference; ab3 = + second difference")
ap.add_argument("--phi", action="store_true", help="residual magnitude profile phi_j(u)/phi_j(u*) in the weights")
ap.add_argument("--phi_prompts", default="512:1536", help="slice of the synthetic prompts used to measure phi")
ap.add_argument("--sbar_file", default="runs/sana512_latents/sbar_N50000.pt", help="within-prompt covariance S_bar and pooled mean m1")
ap.add_argument("--alpha_file", default="runs/sana512_latents/alpha.json", help="calibration coefficient alpha")
ap.add_argument("--keep", action="store_true", help="keep the generated images")
ap.add_argument("--keys_file", default=None, help="generate only the MJHQ keys listed in this file (one per line), for qualitative figures")
ap.add_argument("--no_fid", action="store_true", help="generate only, no FID (with --keep --keys_file)")
ap.add_argument("--out", default="runs/sana_bench")
ap.add_argument("--ref_only", action="store_true", help="compute the FID reference statistics and exit")
ap.add_argument("--tag", default="", help="extra tag in the result key")
A = ap.parse_args()
DEV = "cuda"; t0 = time.time(); el = lambda: f"[{(time.time()-t0)/60:6.1f}m]"
MJ = "data/test/PG-eval-data/MJHQ-30K"
OUT = A.out; os.makedirs(OUT, exist_ok=True)

meta = json.load(open(f"{MJ}/meta_data.json"))
# Must match compute_fid.py, which uses list(data_dict.keys())[:sample_nums] (JSON insertion order, not sorted).
keys = list(meta.keys())[:A.n]
if A.keys_file:
    keys = [k.strip() for k in open(A.keys_file) if k.strip()]
    assert all(k in meta for k in keys), "keys_file contains a key that is not in meta_data.json"
print(f"{el()} MJHQ-30K: {len(meta)} prompts, using {len(keys)}", flush=True)

pipe = SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px_diffusers",
                                    torch_dtype=torch.bfloat16).to(DEV)
pipe.set_progress_bar_config(disable=True)
tr, vae = pipe.transformer, pipe.vae
FACTORY = pipe.scheduler
CHI = inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default
st_b = torch.load(A.sbar_file)
sur = Surrogate("eigen", st_b)                              # within-prompt S_bar in its eigenbasis
m1 = st_b["m1"].to(DEV)                                     # pooled mean
ALPHA = float(json.load(open(A.alpha_file))["alpha"])
print(f"{el()} AREX-C: d={sur.d}  lam_max {float(sur.lam.max()):.2f}  lam_min {float(sur.lam.min()):.2e}  tr {float(sur.lam.sum()):.1f}  alpha {ALPHA:.4f}", flush=True)


def phi_at(u):                                              # (nq,) -> (nq, d), linear interpolation
    i = torch.searchsorted(UG, u.clamp(UG[0], UG[-1]), right=True).clamp(1, len(UG) - 1)
    lo_, hi_ = UG[i - 1], UG[i]; f = ((u - lo_) / (hi_ - lo_).clamp_min(1e-12)).unsqueeze(-1)
    return PHI[i - 1] * (1 - f) + PHI[i] * f
def wk(s, t, wfun, uref):
    """residual weight int_s^t F(t,u) [wfun(u)] [phi(u)/phi(uref)] du; the phi factor only with --phi."""
    if not A.phi: return sur.w_kernel(s, t, wfun)
    def wgt(u):
        base = wfun(u) if wfun is not None else 1.0
        return base * (phi_at(u.squeeze(1).float()) / phi_at(torch.tensor([uref], device=DEV)).clamp_min(1e-20)).double()
    return sur.w_kernel(s, t, wgt)

def save_imgs(imgs, ks, d):
    os.makedirs(d, exist_ok=True)
    for im, k in zip(imgs, ks):
        im.save(os.path.join(d, f"{k}.jpg"), quality=95)

@torch.no_grad()
def decode(lat):
    img = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    img = (img.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((x * 255).round().astype("uint8")) for x in img]

def make_field(ps, steps):
    """Guided velocity field v_w(t, x) for a batch of prompts and the factory time grid t = 1 - sigma."""
    pipe.scheduler = FACTORY
    pipe.scheduler.set_timesteps(steps, device=DEV)
    ts = (1.0 - pipe.scheduler.sigmas.float().cpu().numpy())
    with torch.no_grad():
        pe, pm, ne, nm = pipe.encode_prompt(ps, do_classifier_free_guidance=True, device=DEV,
                                            clean_caption=False, max_sequence_length=300,
                                            complex_human_instruction=CHI)
    emb = torch.cat([ne, pe]).to(tr.dtype); msk = torch.cat([nm, pm]); B = len(ps)
    @torch.no_grad()
    def v(t, xf):
        x = xf.reshape(B, 32, 16, 16)
        o = tr(torch.cat([x, x]).to(tr.dtype), encoder_hidden_states=emb,
               encoder_attention_mask=msk,
               timestep=torch.full((2*B,), (1.0-float(t))*1000.0, device=DEV),
               return_dict=False)[0].float()
        u, c = o.chunk(2)
        return -(u + A.cfg*(c-u)).reshape(B, -1)
    return v, ts

# ---------------------------------------------------------------- residual magnitude profile along the guided ODE
# phi_j(u) = RMS over samples of the residual r_j(u) in S_bar's eigenbasis, along the guided ODE (Euler-50, uniform
# grid) with the AREX-C mean mu(c) = m1 + alpha (x1h(t0) - m1) of each sample.  One profile, shared across prompts.
PHI = UG = None
if A.phi:
    lo, hi = [int(v) for v in A.phi_prompts.split(":")]
    pk = json.load(open("runs/sana512_latents/prompts.json"))[lo:hi]
    STEPS = 50
    ug = torch.linspace(0.0, 1.0, STEPS + 1, device=DEV)[:-1].clamp_min(0.01); h = 1.0 / STEPS
    acc = torch.zeros(STEPS, sur.d, device=DEV, dtype=torch.float64); cnt = 0
    print(f"{el()} measuring phi on synthetic prompts {lo}:{hi} (Euler-{STEPS}, cfg {A.cfg}) ...", flush=True)
    for i0 in range(0, len(pk), A.bs):
        ps_ = pk[i0:i0+A.bs]
        v_, _ = make_field(ps_, STEPS); B_ = len(ps_)
        x = torch.randn(B_, 32, 16, 16, device=DEV, generator=torch.Generator(DEV).manual_seed(9000 + i0)).reshape(B_, -1)
        mu_e = None
        with torch.no_grad():
            for j in range(STEPS):
                u = float(ug[j]); vv = v_(u, x)
                if mu_e is None: mu_e = sur.to_basis(m1 + ALPHA * ((x + (1 - u) * vv) - m1))
                sur.m1e = mu_e; xe = sur.to_basis(x)
                acc[j] += (sur.to_basis(vv) - sur.v_G(u, xe)).double().pow(2).sum(0)
                x = x + h * vv
        cnt += B_
    PHI = (acc / cnt).sqrt().float(); UG = ug
    print(f"{el()} phi measured on {cnt} prompts: ||phi|| at u=0.01/0.5/0.98 = "
          f"{float(PHI[0].norm()):.2f} / {float(PHI[STEPS//2].norm()):.2f} / {float(PHI[-1].norm()):.2f}", flush=True)

# ---------------------------------------------------------------- AREX-C step
def gen_cfsc(ks, ps, seed, steps):
    """AREX-C: affine center mu = m1 + alpha (x1h(t0) - m1), fixed per sample after the first evaluation; covariance
    S_bar.  Residual rule by --scheme: frozen (W0 only), ab2 (+ first Newton difference), ab3 (+ second difference).
    One network evaluation per step."""
    v, ts = make_field(ps, steps)
    B = len(ps)
    x0 = torch.randn(B, 32, 16, 16, device=DEV,
                     generator=torch.Generator(DEV).manual_seed(seed)).reshape(B, -1)
    xe = sur.to_basis(x0); mu_e = None; hist = []           # hist: the last two (u, r); AB3 needs two
    for i in range(len(ts)-1):
        s, t = float(ts[i]), float(ts[i+1])
        u = s if i == 0 else s + A.rho*(t-s)
        if u > s:
            sur.m1e = mu_e; xu = sur.prop(s, u, xe)           # affine transport to the node
        else:
            xu = xe
        x_u = sur.from_basis(xu)
        vv = v(u, x_u)                                        # the one network evaluation of the step
        if mu_e is None:
            mu_e = sur.to_basis(m1 + ALPHA*((x_u + (1-u)*vv) - m1))
        sur.m1e = mu_e
        r = sur.to_basis(vv) - sur.v_G(u, xu)
        step = sur.prop(s, t, xe) + wk(s, t, None, u) * r
        if A.scheme in ("ab2", "ab3") and hist:
            u1, r1 = hist[-1]; f01 = (r - r1) / (u - u1)
            step = step + wk(s, t, lambda w, c=u: w - c, u) * f01
            if A.scheme == "ab3" and len(hist) >= 2:
                u2, r2 = hist[-2]
                f012 = (f01 - (r1 - r2) / (u1 - u2)) / (u - u2)
                step = step + wk(s, t, lambda w, c=u, c1=u1: (w - c) * (w - c1), u) * f012
        xe = step; hist.append((u, r)); hist = hist[-2:]
    return decode(sur.from_basis(xe).reshape(B, 32, 16, 16))

# ---------------------------------------------------------------- FID with SANA's toolkit
SANAROOT = os.path.join(ROOT, "ext", "Sana")
FIDPY = f"{SANAROOT}/tools/metrics/pytorch-fid/compute_fid.py"
REFNPZ = f"{MJ}/MJHQ_30K_512px_fid_embeddings_{A.n}.npz"

def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = ":".join([os.path.abspath(f"{SANAROOT}/tools/metrics/pytorch-fid/src"),
                                os.path.abspath(SANAROOT), e.get("PYTHONPATH", "")])
    return e

def ensure_ref():
    """Reference statistics of the first n MJHQ images (computed once, as compute_fid_embedding.sh does)."""
    if os.path.exists(REFNPZ):
        print(f"{el()} reference statistics found: {REFNPZ}", flush=True); return True
    print(f"{el()} computing reference statistics ({A.n} images) ...", flush=True)
    p = subprocess.run([sys.executable, FIDPY, "--img_size", "512",
                        "--sample_nums", str(A.n), "--path", f"{MJ}/meta_data.json", REFNPZ,
                        "--img_path", f"{MJ}/imgs", "--stat"],
                       capture_output=True, text=True, env=_env())
    ok = os.path.exists(REFNPZ)
    print(f"{el()} reference statistics {'OK' if ok else 'FAILED'}: {(p.stdout+p.stderr)[-400:]}", flush=True)
    return ok

def fid(exp, img_root):
    cmd = [sys.executable, FIDPY, "--img_size", "512",
           "--sample_nums", str(A.n), "--path", REFNPZ, f"{MJ}/meta_data.json",
           "--img_path", img_root, "--exp_name", exp]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.startswith("FID "): return ln
    return f"FID FAILED: {(p.stdout+p.stderr)[-400:]}"

if not A.no_fid and not ensure_ref():
    sys.exit("could not compute the reference statistics")
if A.ref_only:
    print(f"{el()} --ref_only, exiting"); sys.exit(0)

# Results are kept per n and merged, so pilot runs and the 30k runs never overwrite each other.
RES = f"{OUT}/fid_n{A.n}.json"
res = json.load(open(RES)) if os.path.exists(RES) else {}
for n in [int(x) for x in A.nfes.split(",")]:
    tag = "cfsc" + ("" if A.scheme == "frozen" else f"-{A.scheme}") \
                 + ("" if A.rho == 0.5 else f"-rho{A.rho}") \
                 + ("-phi" if A.phi else "") \
                 + (f"-{A.tag}" if A.tag else "") \
                 + (f"-seed{A.seed}" if A.seed else "")
    exp = f"{tag}_nfe{n}"
    root = os.path.join(OUT, "imgs"); d = os.path.join(root, exp)
    txt = os.path.join(root, f"{exp}_sample{A.n}.txt")
    if os.path.exists(txt):
        res[exp] = "FID " + open(txt).read().strip() + f": {exp}"
        print(f"{el()} {exp} already done: {res[exp]}"); continue
    print(f"\n{el()} === {exp} ===", flush=True)
    for i0 in range(0, len(keys), A.bs):
        ks = keys[i0:i0+A.bs]; ps = [meta[k]["prompt"] for k in ks]
        imgs = gen_cfsc(ks, ps, A.seed + i0, n)
        save_imgs(imgs, ks, d)
        if i0 % (A.bs*200) == 0:
            print(f"  {el()} {i0+len(ks)}/{len(keys)}", flush=True)
    pipe.scheduler = FACTORY
    if A.no_fid:
        print(f"{el()} {exp}: generated {len(keys)} images (--no_fid)", flush=True); continue
    line = fid(exp, root)
    res[exp] = line
    print(f"{el()} {exp}: {line}", flush=True)
    if os.path.exists(RES):                                  # re-read before writing: parallel jobs merge
        try: res = {**json.load(open(RES)), **res}
        except Exception: pass
    json.dump(res, open(RES, "w"), indent=1)
    if not A.keep:
        subprocess.run(["rm", "-rf", d])
print(f"\n{el()} === summary ===")
for k, v in res.items(): print(f"  {k:>28}  {v}")
