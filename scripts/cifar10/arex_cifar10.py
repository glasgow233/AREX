"""AREX on CIFAR-10 with the TorchCFM I-CFM checkpoint: FID at 50k samples, NFE 2 to 32.

Sampler (Algorithm 1 of the paper).  The moment-matched affine part is propagated exactly in the eigenbasis of
the target covariance, and the residual r = v_theta - v_G is integrated with the one-node rule ("frozen"), AB2
or AB3, with the residual magnitude profile phi in the quadrature weights.  phi_j(u) is the RMS of the residual
along direction j, measured once along 50-step Euler trajectories of 2048 samples.  The node position rho is
selected per NFE on a held-out split (n_sel samples at seed 777); the reported FID uses seed 1234.

Requires: data/cifar-10-batches-py (FID reference set), ckpt_ext/cfm_cifar10_weights_step_400000.pt and
ext/torchcfm (UNet definition).  The moments m1, Sigma_1 of the flip-augmented training set are computed on
first use and cached in runs/cfm_stats.pt.  Output: runs/arex_cifar10.json.
"""
import os, sys, time, json, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from arex.common import OUT, DEV, TMIN
from arex.surrogate import Surrogate
from arex.cfm import load_cfm, flip_stats
from arex.fid_std import StdFID

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50_000, help="samples for the reported FID")
ap.add_argument("--n-sel", type=int, default=10_000, help="held-out samples for the rho selection")
ap.add_argument("--schemes", default="frozen,ab2,ab3", help="residual rule: frozen (one node) | ab2 | ab3")
ap.add_argument("--nfes", default="2,4,8,16,32")
ap.add_argument("--seeds", default="1234", help="FID seeds; rho is selected once (@777) and reused across seeds")
ap.add_argument("--out", default="arex_cifar10.json", help="results file in runs/ (merged into an existing file)")
ap.add_argument("--rhos", default="0.05,0.1,0.2,0.35,0.5", help="node-position candidates for the held-out selection")
A = ap.parse_args()
t0 = time.time(); el = lambda: f"[{(time.time()-t0)/60:6.1f} min]"

net = load_cfm("cfm", ema=True)
st = flip_stats(os.path.join(OUT, "cfm_stats.pt"))
sur = Surrogate("eigen", st)
print(f"d={sur.d}, lam_max={float(sur.lam.max()):.2f}, lam_min={float(sur.lam.min()):.2e}", flush=True)

@torch.no_grad()
def v_theta(t, x):
    tc = min(max(float(t), TMIN), 1 - TMIN); b = x.shape[0]
    return net(torch.full((b,), tc, device=DEV), x.reshape(b, 3, 32, 32)).reshape(b, -1).float()
def grid(n): return np.linspace(TMIN, 1.0, n + 1)
def noise(b, seed): return torch.randn(b, 3072, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed))

# ---------------------------------------------------------------- residual magnitude profile along the ODE (Euler-50)
@torch.no_grad()
def measure_phi(n=2048, steps=50, seed=23):
    x = noise(n, seed); ug = torch.linspace(TMIN, 1.0, steps + 1, device=DEV)[:-1]
    P = torch.zeros(steps, sur.d, device=DEV); h = (1.0 - TMIN) / steps
    for j in range(steps):
        u = float(ug[j]); v = v_theta(u, x); rr = sur.to_basis(v) - sur.v_G(u, sur.to_basis(x))
        P[j] = rr.pow(2).mean(0).sqrt()
        x = x + h * v
    return ug, P
UG, PHI = measure_phi()
print(f"{el()} phi measured: ||phi|| at u=0.02/0.5/0.98 = {float(PHI[0].norm()):.3f} / {float(PHI[25].norm()):.3f} / {float(PHI[-1].norm()):.3f}", flush=True)
def phi_at(u):                                              # u: tensor (nq,) -> (nq, d), linear interpolation
    i = torch.searchsorted(UG, u.clamp(UG[0], UG[-1]), right=True).clamp(1, len(UG) - 1)
    lo, hi = UG[i - 1], UG[i]; f = ((u - lo) / (hi - lo).clamp_min(1e-12)).unsqueeze(-1)
    return PHI[i - 1] * (1 - f) + PHI[i] * f
def wker(s, t, wfun=None, uref=None, nq=160):
    """int_s^t F(t,u) [phi(u)/phi(uref)] [wfun(u)] du, per direction (160-point trapezoid)."""
    u = torch.linspace(s, t, nq, device=DEV, dtype=torch.float64)[:, None]; l = sur.lam.double()[None, :]
    w = torch.sqrt((1 - t) ** 2 + t ** 2 * l) / torch.sqrt((1 - u) ** 2 + u ** 2 * l)
    if uref is not None:
        w = w * (phi_at(u.squeeze(1).float()) / phi_at(torch.tensor([uref], device=DEV)).clamp_min(1e-20)).double()
    if wfun is not None: w = w * wfun(u)
    return torch.trapezoid(w, u.squeeze(1), dim=0).float()

# ---------------------------------------------------------------- AREX: exact affine transport + residual quadrature
@torch.no_grad()
def sample(scheme, n, nfe, rho, seed, bs=1000):
    ts = grid(nfe); us = [ts[i] + rho * (ts[i+1] - ts[i]) for i in range(nfe)]
    W0 = [wker(ts[i], ts[i+1], uref=us[i]) for i in range(nfe)]
    W1 = [wker(ts[i], ts[i+1], lambda u, c=us[i]: u - c, uref=us[i]) for i in range(nfe)]
    W2 = [wker(ts[i], ts[i+1], lambda u, c=us[i], c1=us[i-1]: (u-c)*(u-c1), uref=us[i]) if i >= 2 else None for i in range(nfe)]
    out = []
    for i0 in range(0, n, bs):
        xe = sur.to_basis(noise(min(bs, n - i0), seed + i0)); hist = []
        for i in range(nfe):
            xu = sur.prop(ts[i], us[i], xe)                                            # affine transport to the node
            r = sur.to_basis(v_theta(us[i], sur.from_basis(xu))) - sur.v_G(us[i], xu)  # the one network evaluation
            step = sur.prop(ts[i], ts[i+1], xe) + W0[i] * r
            if scheme in ("ab2", "ab3") and hist:                                     # Newton form of AB2 / AB3
                u1, r1 = hist[-1]
                f01 = (r - r1) / (us[i] - u1); step = step + W1[i] * f01
                if scheme == "ab3" and len(hist) >= 2 and W2[i] is not None:
                    u2, r2 = hist[-2]
                    step = step + W2[i] * ((f01 - (r1 - r2) / (u1 - u2)) / (us[i] - u2))
            xe = step; hist.append((us[i], r)); hist = hist[-2:]
        out.append(sur.from_basis(xe))
    return torch.cat(out)

# ---------------------------------------------------------------- held-out selection of rho, then FID
fid = StdFID(); NFES = [int(x) for x in A.nfes.split(",")]; RHOS = tuple(float(x) for x in A.rhos.split(","))
SEEDS = [int(x) for x in A.seeds.split(",")]
OUTF = os.path.join(OUT, A.out)
res = json.load(open(OUTF)) if os.path.exists(OUTF) else {"fid": {}, "rho": {}}
res["phi_norm"] = [float(x) for x in PHI.norm(dim=1)]; res["u"] = [float(x) for x in UG]
def dump(): json.dump(res, open(OUTF, "w"), indent=1)
for scheme in A.schemes.split(","):
    assert scheme in ("frozen", "ab2", "ab3"), scheme
    print(f"\n{el()} === {scheme}: rho selection n={A.n_sel} @ 777 ===", flush=True)
    res["rho"].setdefault(scheme, {}); rows = res["fid"].setdefault(scheme, {})
    for nfe in NFES:
        if str(nfe) in res["rho"][scheme]: continue          # selected in an earlier run: reuse
        sc = {r: fid(sample(scheme, A.n_sel, nfe, r, 777)) for r in RHOS}; rb = min(sc, key=sc.get)
        res["rho"][scheme][str(nfe)] = rb; dump()
        print(f"  NFE={nfe:>3}  " + "  ".join(f"r={r}:{sc[r]:6.2f}" for r in RHOS) + f"  -> {rb}", flush=True)
    for nfe in NFES:
        for seed in SEEDS:
            key = str(nfe) if seed == 1234 else f"{nfe}@{seed}"
            if key in rows: continue
            rows[key] = fid(sample(scheme, A.n, nfe, res["rho"][scheme][str(nfe)], seed)); dump()
            print(f"  {scheme} NFE={nfe:>3} seed {seed}  FID {rows[key]:7.2f}   {el()}", flush=True)
print(f"\n  {'sampler':<10}" + "".join(f"{'NFE='+str(n):>10}" for n in NFES))
for k in res["fid"]:
    print(f"  {k:<10}" + "".join(f"{res['fid'][k][str(n)]:>10.2f}" if str(n) in res["fid"][k] else f"{'-':>10}" for n in NFES))
print(f"\n{el()} done -> {OUTF}")
