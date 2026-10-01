"""AREX-C moments for SiT-XL/2 (cfg 1.5) from the FULL ImageNet-1k training set (1.28M images + flips).

Source: benjamin-paine/imagenet-1k-256x256 (ILSVRC-2012 train at 256^2), sd-vae-ft-mse encoder,
latent_dist.sample() * 0.18215, horizontal flips.  Nothing is stored: the moments are accumulated on the fly (total scatter G = sum z z^T in float64, per-class sums and counts), so
  m1        = sum z / N
  Sigma_1   = (G - N m1 m1^T) / (N - 1)
  S_bar     = (G - sum_c n_c mu_c mu_c^T) / (N - 1000)      pooled within-class covariance
  mu(c)     = per-class mean
alpha is then the same least squares as before against the guided network's endpoint predictions.
Shards are downloaded one at a time and deleted.  JPEG decoding runs in DataLoader workers.
Requires ckpt_ext/SiT-XL-2-256.pt, ckpt_ext/sd-vae-ft-mse and ext/sit (models.py).
Output: runs/sit_data_full_stats.pt / .json (the --stats file of arex_sit.py).  Timing of the encoder pass is printed separately."""
import os, sys, io, time, json, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "ext", "sit"))
ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=40, help="number of parquet shards of the dataset")
ap.add_argument("--max-shards", type=int, default=0, help="process only the first k shards (quick check); 0 = all")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--tmp", default=os.path.join(ROOT, "runs", "imagenet_tmp"))
A = ap.parse_args()
from models import SiT_models
from arex.common import OUT, DEV, CKPT_DIR
from diffusers.models import AutoencoderKL
from PIL import Image
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

CK = os.path.join(CKPT_DIR, "SiT-XL-2-256.pt")
TMIN, W, NULL, BS = 1e-3, 1.5, 1000, 125
t00 = time.time(); el = lambda: f"[{(time.time()-t00)/60:6.1f} min]"
d = 4 * 32 * 32
REPO = "benjamin-paine/imagenet-1k-256x256"

class Rows(Dataset):
    def __init__(self, imgs, labels): self.imgs, self.labels = imgs, labels
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        im = Image.open(io.BytesIO(self.imgs[i]["bytes"])).convert("RGB")
        if im.size != (256, 256): im = im.resize((256, 256), Image.LANCZOS)
        return torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1), int(self.labels[i])

vae = AutoencoderKL.from_pretrained(os.path.join(CKPT_DIR, "sd-vae-ft-mse")).to(DEV).eval()
g_vae = torch.Generator(device=DEV).manual_seed(0)
G = torch.zeros(d, d, dtype=torch.float64, device=DEV)      # total scatter
S = torch.zeros(1000, d, dtype=torch.float64, device=DEV)   # per-class sums
n = torch.zeros(1000, dtype=torch.float64, device=DEV)
N = 0; t_enc = 0.0; t_acc = 0.0
os.makedirs(A.tmp, exist_ok=True)
for si in range(min(A.shards, A.max_shards) if A.max_shards else A.shards):
    fn = f"data/train-{si:05d}-of-{A.shards:05d}.parquet"
    p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=A.tmp)
    tab = pq.read_table(p, columns=["image", "label"])
    ds = Rows(tab.column("image").to_pylist(), tab.column("label").to_numpy())
    dl = DataLoader(ds, batch_size=A.bs, num_workers=A.workers, pin_memory=True)
    with torch.no_grad():
        for u8, y in dl:
            x = u8.to(DEV, non_blocking=True).float() / 127.5 - 1.0
            x = torch.cat([x, x.flip(-1)]); yy = torch.cat([y, y]).to(DEV)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encode(x).latent_dist.sample(generator=g_vae) * 0.18215
            z = z.float().reshape(len(x), -1)
            torch.cuda.synchronize(); t_enc += time.time() - t0; t0 = time.time()
            zd = z.double(); G += zd.T @ zd
            S.index_add_(0, yy, zd); n.index_add_(0, yy, torch.ones(len(yy), dtype=torch.float64, device=DEV))
            N += len(z)
            torch.cuda.synchronize(); t_acc += time.time() - t0
    del tab, ds, dl; os.remove(p)
    print(f"{el()} shard {si}: {N} latents so far   encoder {t_enc/60:.1f} min   accumulation {t_acc/60:.1f} min", flush=True)
print(f"{el()} {N} latents from {N//2} images; VAE encoding {t_enc/60:.1f} min ({1000*t_enc/N:.2f} ms/encode), "
      f"moment accumulation {t_acc/60:.1f} min", flush=True)
del vae; torch.cuda.empty_cache()

# ---------------------------------------------------------------- moments
m1 = (S.sum(0) / N)
present = n > 0; K = int(present.sum())              # K = 1000 on the full training set; fewer with --max-shards
if K < 1000: print(f"  NOTE: only {K}/1000 classes present; absent classes get mu(c) = m1 (quick check only)", flush=True)
mu_c = torch.where(present[:, None], S / n.clamp_min(1)[:, None], m1[None, :])
Sig1 = ((G - N * torch.outer(m1, m1)) / (N - 1)).cpu()
Sbar = ((G - (mu_c.T * n) @ mu_c) / (N - K)).cpu()
Cov_mu = ((mu_c[present] - m1).T @ (mu_c[present] - m1) / max(K - 1, 1)).cpu()
print(f"{el()} tr Sigma_1 = {Sig1.trace():.1f}   tr S_bar = {Sbar.trace():.1f}   tr Cov_c(mu) = {Cov_mu.trace():.1f}"
      f"   between-class share = {100*Cov_mu.trace()/(Sbar.trace()+Cov_mu.trace()):.1f}%   n_c min/max {int(n.min())}/{int(n.max())}", flush=True)
t0 = time.time()
def eig(C):
    l, U = torch.linalg.eigh(0.5 * (C + C.T).to(DEV)); return l.clamp_min(1e-10).cpu(), U.cpu()
lam_p, U_p = eig(Sig1); lam_b, U_b = eig(Sbar); torch.cuda.synchronize()
print(f"{el()} two eigendecompositions: {time.time()-t0:.2f} s", flush=True)
def kappa(lam, t=0.5): s = torch.sqrt((1-t)**2 + t*t*lam); return float(s.max() / s.min())
def summ(name, lam):
    lam = lam.flip(0)
    print(f"  {name:<28} lam_max {lam[0]:8.3f}  median {lam[lam.numel()//2]:.4f}  #>1 {(lam>1).sum().item():4d}/{lam.numel()}"
          f"  kappa(0.5) {kappa(lam):6.2f}  std/mean {lam.std()/lam.mean():.2f}", flush=True)
    return dict(lam_max=float(lam[0]), median=float(lam[lam.numel()//2]), n_above_1=int((lam>1).sum()), kappa=kappa(lam),
                std_over_mean=float(lam.std()/lam.mean()))
stats = {"w": W, "source": f"ImageNet train, {'all' if not A.max_shards else f'first {A.max_shards}/{A.shards} shards of the'} images + flips ({REPO}), N={N}",
         "N_latents": N, "encode_min": t_enc / 60, "ms_per_encode": 1000 * t_enc / N, "accumulate_min": t_acc / 60,
         "between_share": float(Cov_mu.trace()/(Sbar.trace()+Cov_mu.trace())),
         "tr": {"Sigma1": float(Sig1.trace()), "Sbar": float(Sbar.trace()), "Cov_mu": float(Cov_mu.trace())}}
stats["pooled_data"] = summ("pooled Sigma_1 (data, full)", lam_p)
stats["sbar_data"] = summ("within-class S_bar (data, full)", lam_b)

# ---------------------------------------------------------------- alpha
net = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000).to(DEV).eval()
sd = torch.load(CK, map_location="cpu"); net.load_state_dict(sd.get("ema", sd)); del sd
for p_ in net.parameters(): p_.requires_grad_(False)
@torch.no_grad()
def v_w(t, x, y, w=W):
    b = len(x); tc = min(max(float(t), TMIN), 1 - TMIN)
    xx = torch.cat([x, x]).view(2 * b, 4, 32, 32); yy = torch.cat([y, torch.full_like(y, NULL)])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = net(xx, torch.full((2 * b,), tc, device=DEV), yy).float()
    vc, vu = o.chunk(2)
    eps = vu[:, :3] + w * (vc[:, :3] - vu[:, :3])
    return torch.cat([eps, vc[:, 3:]], dim=1).reshape(b, -1)
print(f"\n{el()} alpha calibration (one x0 per class, endpoint prediction at t0={TMIN})", flush=True)
t0 = time.time()
m1d = m1.float(); A_ = torch.zeros(1000, d, device=DEV)
g = torch.Generator(device=DEV).manual_seed(77)
for c0 in range(0, 1000, BS):
    y = torch.arange(c0, c0 + BS, device=DEV); x0 = torch.randn(BS, d, device=DEV, generator=g)
    xt = (1 - TMIN) * x0
    A_[c0:c0+BS] = xt + (1 - TMIN) * v_w(TMIN, xt, y) - m1d
B_ = (mu_c.float() - m1d)
def ls(idx): return float((A_[idx] * B_[idx]).sum() / (A_[idx] ** 2).sum())
a_lo, a_hi, a_all = ls(slice(0, 500)), ls(slice(500, 1000)), ls(slice(0, 1000))
torch.cuda.synchronize()
print(f"  alpha classes 0-499 {a_lo:.4f}   500-999 {a_hi:.4f}   all {a_all:.4f}   ({time.time()-t0:.1f} s, 1000 evaluations)", flush=True)
stats["alpha"] = {"lo": a_lo, "hi": a_hi, "all": a_all}
torch.save({"lam_pool": lam_p.float(), "U_pool": U_p.float(), "lam_bar": lam_b.float(), "U_bar": U_b.float(),
            "m1": m1.float().cpu(), "mu_c": mu_c.float().cpu(), "alpha": a_all, "w": W, "source": stats["source"]},
           os.path.join(OUT, "sit_data_full_stats.pt"))
json.dump(stats, open(os.path.join(OUT, "sit_data_full_stats.json"), "w"), indent=1)
print(f"\n{el()} done -> runs/sit_data_full_stats.pt / .json")
