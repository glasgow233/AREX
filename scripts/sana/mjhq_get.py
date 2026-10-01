"""Download MJHQ-30K (prompts and FID reference images of the SANA / STORK protocol).

The SANA metric toolkit expects data/test/PG-eval-data/MJHQ-30K/{meta_data.json, imgs/}.
The dataset is gated on Hugging Face: accept its terms and log in (huggingface-cli login) first.
"""
import os, json, time, sys
t0 = time.time(); el = lambda: f"[{(time.time()-t0)/60:5.1f}m]"
from huggingface_hub import snapshot_download
DST = "data/test/PG-eval-data/MJHQ-30K"
os.makedirs(DST, exist_ok=True)
try:
    p = snapshot_download(repo_id="playgroundai/MJHQ-30K", repo_type="dataset", local_dir=DST, max_workers=8)
    print(f"{el()} downloaded to {p}", flush=True)
except Exception as e:
    print(f"{el()} failed: {type(e).__name__}: {e}", flush=True)
    print("  a 401/403 means the dataset terms have not been accepted or no token is configured", flush=True)
    sys.exit(1)
for root, dirs, files in os.walk(DST):
    if root.count(os.sep) - DST.count(os.sep) > 1: continue
    print(f"  {root}: {len(files)} files, {len(dirs)} directories")
mj = os.path.join(DST, "meta_data.json")
if os.path.exists(mj):
    m = json.load(open(mj))
    print(f"\n{el()} meta_data.json: {len(m)} entries")
