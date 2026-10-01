"""AREX-C on SiT-XL/2 (ImageNet-256, class-conditional, classifier-free guidance w = 1.5): FID, sFID, Inception
score, precision and recall at 50k samples with the conventions of the ADM evaluator.

Guided field: v_w = v_null + w (v_c - v_null) on the first three latent channels (SiT's forward_with_cfg); one
double-batch forward = 1 NFE.  Time grid linspace(TMIN, 1, nfe + 1).  Labels arange(n) % 1000, noise seed
1234 + batch offset, batch 125.

AREX-C (Algorithm 2 of the paper).  The within-class covariance S_bar and the calibration coefficient alpha come
from the moment file in runs/ (--stats), computed from the ImageNet-1k training set by sit_moments.py.  The first
network evaluation is at t0 and gives the affine center mu(c) = m1 + alpha (x1h - m1) per sample; later nodes sit
at u* = s + rho (t - s).  With --phi the residual weights carry the residual magnitude profile, measured once along
guided Euler-50 trajectories.  rho is selected per NFE on a held-out split (n_sel samples at seed 777).

Requires: ckpt_ext/SiT-XL-2-256.pt, ckpt_ext/sd-vae-ft-mse/, runs/VIRTUAL_imagenet256_labeled.npz (the ADM
reference batch) and ext/sit (models.py).  Output: runs/sit_cfg_<method>[_phi][tag].json.
"""
import os, sys, time, json, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "ext", "sit"))
from models import SiT_models
from diffusers.models import AutoencoderKL
from arex.common import OUT, DEV, CKPT_DIR
from arex.fid_inception import InceptionV3, fid_inception_v3
from arex.fid_std import frechet_distance, stats_of

CK = os.path.join(CKPT_DIR, "SiT-XL-2-256.pt")
VAE_DIR = os.path.join(CKPT_DIR, "sd-vae-ft-mse")
REF_NPZ = os.path.join(OUT, "VIRTUAL_imagenet256_labeled.npz")
TMIN, SCALE, BS, BSV, NULL = 1e-3, 0.18215, 125, 50, 1000

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["cfsc", "cfsc_ab2", "cfsc_ab3"],
                help="AREX-C with the one-node rule (cfsc), AB2 (cfsc_ab2) or AB3 (cfsc_ab3); cfsc is the legacy name of AREX-C")
ap.add_argument("--w", type=float, default=1.5)
ap.add_argument("--n", type=int, default=50_000)
ap.add_argument("--n-sel", type=int, default=2_500)
ap.add_argument("--dump", default="", help="qualitative mode: save decoded PNGs for --classes at the config selected in the results json, no FID")
ap.add_argument("--classes", default="207,250,285,388,933,979,980,812,417,963,291,22,130,340,386,975")
ap.add_argument("--dump-seed", type=int, default=1234)
ap.add_argument("--nfes", default="2,4,8,16,32")
ap.add_argument("--rho-fixed", type=float, default=None, help="skip the held-out selection and use this rho")
ap.add_argument("--tag", default="", help="suffix for the results json")
ap.add_argument("--stats", default="sit_data_full_stats.pt", help="moment file in runs/ (output of sit_moments.py)")
ap.add_argument("--seed", type=int, default=1234, help="noise seed of the evaluation run (selection stays at 777)")
ap.add_argument("--rho-from", default="", help="results json in runs/ whose per-NFE _cfg is reused instead of a selection")
ap.add_argument("--phi", action="store_true",
                help="scale the residual weights by the per-direction residual-magnitude profile phi_j(u)/phi_j(u*), "
                     "measured once along guided Euler-50 trajectories with the AREX-C mean")
A = ap.parse_args()
NFES = [int(v) for v in A.nfes.split(",")]
t00 = time.time(); el = lambda: f"[{(time.time()-t00)/60:6.1f} min]"

net = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000).to(DEV).eval()
sd = torch.load(CK, map_location="cpu"); net.load_state_dict(sd.get("ema", sd)); del sd
for p in net.parameters(): p.requires_grad_(False)
vae = AutoencoderKL.from_pretrained(VAE_DIR).to(DEV).eval()
for p in vae.parameters(): p.requires_grad_(False)
inc = InceptionV3([2, 3], resize_input=True, normalize_input=True, use_fid_inception=True).to(DEV).eval()
d = 4 * 32 * 32

CALLS = [0]
@torch.no_grad()
def v_w(t, x, y):
    CALLS[0] += 1
    b = len(x); tc = min(max(float(t), TMIN), 1 - TMIN)
    xx = torch.cat([x, x]).view(2 * b, 4, 32, 32); yy = torch.cat([y, torch.full_like(y, NULL)])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = net(xx, torch.full((2 * b,), tc, device=DEV), yy).float()
    vc, vu = o.chunk(2)
    eps = vu[:, :3] + A.w * (vc[:, :3] - vu[:, :3])
    return torch.cat([eps, vc[:, 3:]], dim=1).reshape(b, -1)

def grid(n): return np.linspace(TMIN, 1.0, n + 1)
def noise(b, seed): return torch.randn(b, d, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed))

# ---------------------------------------------------------------- AREX-C: moments, affine propagator, weights
ST = torch.load(os.path.join(OUT, A.stats), map_location=DEV)
print(f"{el()} moments from runs/{A.stats}" + (f" ({ST['source']})" if "source" in ST else ""), flush=True)
lam, U, m1, ALPHA = ST["lam_bar"].float(), ST["U_bar"].float(), ST["m1"].float(), float(ST["alpha"])
assert abs(float(ST["w"]) - A.w) < 1e-9, "stats were computed at a different guidance scale"
print(f"{el()} AREX-C: S_bar lam_max {float(lam.max()):.2f}  lam_min {float(lam.min()):.2e}  tr {float(lam.sum()):.1f}  alpha {ALPHA:.4f}  w {A.w}", flush=True)
sig = lambda t: torch.sqrt((1 - t) ** 2 + t ** 2 * lam)
kfn = lambda t: (-(1 - t) + t * lam) / sig(t) ** 2
prop = lambda s, t, xe, me: t * me + (sig(t) / sig(s)) * (xe - s * me)
vG = lambda t, xe, me: me + kfn(t) * (xe - t * me)
def wker(s, t, wfun=None, uref=None, nq=160):
    """int_s^t F(t,u) [phi(u)/phi(uref)] [wfun(u)] du, per direction (160-point trapezoid)."""
    u = torch.linspace(s, t, nq, device=DEV, dtype=torch.float64)[:, None]; l = lam.double()[None, :]
    w = torch.sqrt((1 - t) ** 2 + t ** 2 * l) / torch.sqrt((1 - u) ** 2 + u ** 2 * l)
    if A.phi and uref is not None:
        w = w * (phi_at(u.squeeze(1).float()) / phi_at(torch.tensor([uref], device=DEV)).clamp_min(1e-20)).double()
    if wfun is not None: w = w * wfun(u)
    return torch.trapezoid(w, u.squeeze(1), dim=0).float()

# phi_j(u) = RMS over samples of the residual r_j(u) in S_bar's eigenbasis, along the guided ODE (Euler-50, uniform
# grid from TMIN) with each sample's AREX-C mean mu(c) = m1 + alpha (x1h(t0) - m1) from the first evaluation.
PHI = UG = None
@torch.no_grad()
def measure_phi(n=2048, steps=50, seed=23):
    ug = torch.linspace(TMIN, 1.0, steps + 1, device=DEV)[:-1]; h = (1.0 - TMIN) / steps
    acc = torch.zeros(steps, d, device=DEV, dtype=torch.float64)
    for i0 in range(0, n, BS):
        y = (torch.arange(i0, i0 + BS, device=DEV) * 7919) % 1000       # spread over classes, independent of eval labels
        x = noise(BS, seed + i0); me = None
        for j in range(steps):
            u = float(ug[j]); vv = v_w(u, x, y)
            if me is None: me = (m1 + ALPHA * ((x + (1 - u) * vv) - m1)) @ U
            rr = ((vv @ U) - vG(u, x @ U, me)).double(); acc[j] += rr.pow(2).sum(0)
            x = x + h * vv
    return ug, (acc / n).sqrt().float()
if A.phi:
    UG, PHI = measure_phi()
    print(f"{el()} phi measured (2048 guided Euler-50 trajectories): ||phi|| at u=0.001/0.5/0.98 = "
          f"{float(PHI[0].norm()):.2f} / {float(PHI[25].norm()):.2f} / {float(PHI[-1].norm()):.2f}", flush=True)
def phi_at(u):                                              # (nq,) -> (nq, d), linear interpolation
    i = torch.searchsorted(UG, u.clamp(UG[0], UG[-1]), right=True).clamp(1, len(UG) - 1)
    lo, hi = UG[i - 1], UG[i]; f = ((u - lo) / (hi - lo).clamp_min(1e-12)).unsqueeze(-1)
    return PHI[i - 1] * (1 - f) + PHI[i] * f

@torch.no_grad()
def sample_cfsc(scheme, rho, y, nfe, seed):
    ts = grid(nfe); x0 = noise(len(y), seed); xe = x0 @ U; me = None; hist = []
    for i in range(nfe):
        s, t = float(ts[i]), float(ts[i+1])
        u = s if i == 0 else s + rho * (t - s)                # the first node is the grid point t0
        xu = xe if i == 0 else prop(s, u, xe, me)             # affine transport to the node
        vv = v_w(u, xu @ U.T, y)                              # the one network evaluation of the step
        if me is None:                                        # first evaluation: affine center per sample
            me = (m1 + ALPHA * ((xu @ U.T + (1 - u) * vv) - m1)) @ U
        r = (vv @ U) - vG(u, xu, me)
        step = prop(s, t, xe, me) + wker(s, t, uref=u) * r
        if scheme in ("ab2", "ab3") and hist:                 # Newton form of AB2 / AB3
            u1, r1 = hist[-1]; f01 = (r - r1) / (u - u1); step = step + wker(s, t, lambda q, c=u: q - c, uref=u) * f01
            if scheme == "ab3" and len(hist) >= 2:
                u2, r2 = hist[-2]; f012 = (f01 - (r1 - r2) / (u1 - u2)) / (u - u2)
                step = step + wker(s, t, lambda q, c=u, c1=u1: (q - c) * (q - c1), uref=u) * f012
        xe = step; hist.append((u, r)); hist = hist[-2:]
    return xe @ U.T

SCHEME = {"cfsc": "frozen", "cfsc_ab2": "ab2", "cfsc_ab3": "ab3"}
def sampler(cfg):
    return lambda y, nfe, seed: sample_cfsc(SCHEME[cfg[0]], cfg[1], y, nfe, seed)

# ---------------------------------------------------------------- ADM metrics
FCW = fid_inception_v3().fc.to(DEV).eval(); TAP = {}
inc.blocks[2][6].branch1x1.register_forward_hook(lambda m, i, o: TAP.__setitem__("sp", o))
z = np.load(REF_NPZ); MU_R, SIG_R, MU_S, SIG_S = z["mu"], z["sigma"], z["mu_s"], z["sigma_s"]
@torch.no_grad()
def acts(u8):
    _, pool = inc(u8.float() / 255.0); pool = pool.squeeze(-1).squeeze(-1)
    return pool, TAP["sp"][:, :7].permute(0, 2, 3, 1).reshape(len(u8), -1), FCW(pool)
@torch.no_grad()
def ref_pool():
    """pool3 features of the reference images, for precision / recall (cached)."""
    c = os.path.join(OUT, "imagenet256_ref_pool.npy")
    if os.path.isfile(c): return np.load(c)
    out = []
    for i in range(0, len(z["arr_0"]), BSV):
        q = torch.from_numpy(z["arr_0"][i:i+BSV]).to(DEV).permute(0, 3, 1, 2).contiguous()
        out.append(acts(q)[0].cpu())
    a = torch.cat(out).numpy(); np.save(c, a); return a
REF_POOL = ref_pool()
def _radii(A_, k=3, ch=2048):
    n = len(A_); r = torch.empty(n, device=DEV)
    for i in range(0, n, ch):
        dd = torch.cdist(A_[i:i+ch], A_); dd[torch.arange(len(dd), device=DEV), torch.arange(i, min(i+ch, n), device=DEV)] = float("inf")
        r[i:i+ch] = dd.topk(k, largest=False).values[:, -1]
    return r
def _cov(B, A_, rA, ch=2048):
    h = torch.zeros(len(B), dtype=torch.bool, device=DEV)
    for i in range(0, len(B), ch): h[i:i+ch] = (torch.cdist(B[i:i+ch], A_) <= rA[None, :]).any(1)
    return float(h.float().mean())
def prec_recall(fp):
    R = torch.from_numpy(REF_POOL).to(DEV).float(); F_ = torch.from_numpy(fp).to(DEV).float()
    return _cov(F_, R, _radii(R)), _cov(R, F_, _radii(F_))
def inception_score(lg, splits=10):
    p = torch.softmax(torch.from_numpy(lg).float(), 1); s = []
    for c in p.chunk(splits):
        py = c.mean(0, keepdim=True); s.append(float((c * (c.clamp_min(1e-12).log() - py.clamp_min(1e-12).log())).sum(1).mean().exp()))
    return float(np.mean(s))
@torch.no_grad()
def decode(zz):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        img = vae.decode(zz.view(-1, 4, 32, 32) / SCALE).sample
    return ((img.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
@torch.no_grad()
def evaluate(fn, n, nfe, seed=1234, full=True):
    ya = torch.arange(n, device=DEV) % 1000; P, S, L = [], [], []; CALLS[0] = 0
    for i0 in range(0, n, BS):
        zz = fn(ya[i0:i0+BS], nfe, seed + i0)
        if not torch.isfinite(zz).all(): return dict(fid=float("inf"))
        for j in range(0, len(zz), BSV):
            a, b, c = acts(decode(zz[j:j+BSV])); P.append(a.double().cpu())
            if full: S.append(b.double().cpu()); L.append(c.float().cpu())
    per = CALLS[0] / ((n + BS - 1) // BS); assert abs(per - nfe) < 1e-9, f"NFE accounting off: {per} at nfe={nfe}"
    P = torch.cat(P).numpy(); mu, sg = stats_of(P); out = dict(fid=frechet_distance(mu, sg, MU_R, SIG_R))
    if full:
        ms, ss = stats_of(torch.cat(S).numpy()); out["sfid"] = frechet_distance(ms, ss, MU_S, SIG_S)
        out["is"] = inception_score(torch.cat(L).numpy()); out["prec"], out["rec"] = prec_recall(P.astype(np.float32))
    return out

# ---------------------------------------------------------------- selection + evaluation
CANDS = {"cfsc":     [("cfsc", r) for r in (0.2, 0.35, 0.5)],
         "cfsc_ab2": [("cfsc_ab2", r) for r in ((0.05, 0.1, 0.2, 0.35) if A.phi else (0.05, 0.1, 0.2))],
         "cfsc_ab3": [("cfsc_ab3", r) for r in ((0.05, 0.1, 0.2, 0.35) if A.phi else (0.05, 0.1, 0.2))]}[A.method]
if A.rho_fixed is not None: CANDS = [(A.method, A.rho_fixed)]
RES = os.path.join(OUT, f"sit_cfg_{A.method}{'_phi' if A.phi else ''}{A.tag}{'' if A.seed == 1234 else f'_seed{A.seed}'}.json")
if A.dump:
    from PIL import Image
    os.makedirs(A.dump, exist_ok=True)
    sel = json.load(open(RES)).get("_cfg", {}) if os.path.exists(RES) else {}
    cls = [int(c) for c in A.classes.split(",")]; y = torch.tensor(cls, device=DEV)
    for nfe in NFES:
        cfg = tuple(sel[str(nfe)]) if str(nfe) in sel else CANDS[0]
        zz = sampler(cfg)(y, nfe, A.dump_seed)
        u8 = decode(zz).permute(0, 2, 3, 1).cpu().numpy()
        for c, im in zip(cls, u8):
            Image.fromarray(im).save(os.path.join(A.dump, f"{A.method}_nfe{nfe}_c{c:03d}.png"))
        print(f"{el()} dumped {A.method} NFE {nfe} cfg={cfg} -> {A.dump}", flush=True)
    sys.exit(0)

best = {}
if A.rho_from:
    cf = json.load(open(os.path.join(OUT, A.rho_from)))["_cfg"]
    best = {nfe: tuple(cf[str(nfe)]) for nfe in NFES}
    assert all(best[nfe][0] == A.method for nfe in NFES), "the --rho-from file belongs to another method"
    print(f"\n{el()} === {A.method}: configurations reused from {A.rho_from}: {best} ===", flush=True)
elif len(CANDS) > 1:
    print(f"\n{el()} === {A.method}: selection, held-out n={A.n_sel} @ seed 777 ===", flush=True)
    for nfe in NFES:
        sc = {c: evaluate(sampler(c), A.n_sel, nfe, 777, full=False)["fid"] for c in CANDS}
        best[nfe] = min(sc, key=sc.get)
        print(f"  NFE={nfe:>3}  " + "  ".join(f"{c}:{sc[c]:7.2f}" for c in CANDS) + f"   -> {best[nfe]}", flush=True)
else:
    best = {nfe: CANDS[0] for nfe in NFES}
print(f"\n{el()} === {A.method}: evaluation n={A.n} @ seed {A.seed}, cfg {A.w} ===", flush=True)
out = {"_cfg": {str(k): list(v) for k, v in best.items()}, "_w": A.w, "_stats": A.stats, "_seed": A.seed}
for nfe in NFES:
    out[nfe] = evaluate(sampler(best[nfe]), A.n, nfe, seed=A.seed); m = out[nfe]
    print(f"  NFE={nfe:>4} cfg={best[nfe]}  FID {m['fid']:7.3f}  sFID {m['sfid']:7.3f}  IS {m['is']:7.1f}  P {m['prec']:.3f}  R {m['rec']:.3f}   {el()}", flush=True)
    json.dump(out, open(RES, "w"), indent=1)
print(f"\n{el()} done -> {RES}")
