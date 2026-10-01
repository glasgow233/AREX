"""TorchCFM CIFAR-10 checkpoints: network loader and target moments.

Unlike edm_adapter / ssde_adapter, there is NOTHING to reparameterise here.
TorchCFM's CIFAR-10 models are trained (train_cifar10.py) as

    x1 = data in [-1,1]        (RandomHorizontalFlip augmented)
    x0 = torch.randn_like(x1)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    loss = mean((net(t, xt) - ut)**2)

with sigma = 0.0, so for ConditionalFlowMatcher (`cfm`, I-CFM):
    xt = t*x1 + (1-t)*x0 ,  ut = x1 - x0
which is EXACTLY our interpolant X_t = (1-t)X0 + t X1, with t=0 noise and
t=1 data, and the network output IS the velocity.  Their own sampler
(utils_cifar.generate_samples) integrates t: 0 -> 1 from randn, confirming
the direction.  So v_theta(t,x) = net(t, x), full stop.

Three released checkpoints, all at 400k steps:
  cfm    ConditionalFlowMatcher                       independent coupling
  fm     TargetConditionalFlowMatcher                 independent coupling,
         xt = t*x1 + (1-t)*eps with eps a FRESH draw -- same marginal path and
         hence the same marginal velocity field as icfm, different conditional
         target (so: a variance-reduction ablation, not a different model class)
  otcfm  ExactOptimalTransportConditionalFlowMatcher  MINIBATCH-OT coupling
         (batch 128).  Same endpoint marginals, but (X0,X1) are now dependent,
         so Cov(X_t) = (1-t)^2 I + t^2 Sigma1 + 2t(1-t) Z with Z = Cov(X0,X1).
         Our v_G assumes Z = 0, so otcfm is a MISMATCHED (harder) case for us.
         Best published FID of the three (3.5), so it is the headline backbone
         and the mismatch has to be measured, not assumed away.

Because the model is trained on t ~ U[0,1] all the way to t=0, there is no
untrained region and hence no T0 skip of the kind edm_adapter needs.
"""
import os, sys, torch
from arex.common import ROOT, DEV, CKPT_DIR, load_cifar

TCFM_REPO = os.path.join(ROOT, "ext", "torchcfm")
CKPT = {k: os.path.join(CKPT_DIR, f"{k}_cifar10_weights_step_400000.pt")
        for k in ("otcfm", "cfm", "fm")}


def _stub_ot():
    """torchcfm/__init__.py imports conditional_flow_matching -> optimal_transport
    -> `import ot as pot` (POT), which is not installed here.  We only ever need
    the UNet, and every POT reference in that file is inside a function body, so
    a lazy stub is enough to let the import chain complete.  It raises if it is
    ever actually reached, so this can never silently change a result.
    (Where we do need an OT plan -- measuring the minibatch-OT cross-covariance
    -- we compute it ourselves with scipy's exact linear_sum_assignment, which
    for equal-mass n-vs-n is exactly what pot.emd returns.)"""
    import types
    if "ot" in sys.modules:
        return
    def _boom(name):
        def f(*a, **k):
            raise RuntimeError(f"POT is stubbed out but ot.{name} was actually called")
        return f
    m = types.ModuleType("ot")
    m.__getattr__ = _boom
    for sub in ("unbalanced", "partial"):
        s_ = types.ModuleType(f"ot.{sub}")
        s_.__getattr__ = _boom
        setattr(m, sub, s_); sys.modules[f"ot.{sub}"] = s_
    sys.modules["ot"] = m


def load_cfm(which="cfm", ema=True):
    if TCFM_REPO not in sys.path:
        sys.path.insert(0, TCFM_REPO)
    _stub_ot()
    from torchcfm.models.unet.unet import UNetModelWrapper
    net = UNetModelWrapper(dim=(3, 32, 32), num_res_blocks=2, num_channels=128,
                           channel_mult=[1, 2, 2, 2], num_heads=4,
                           num_head_channels=64, attention_resolutions="16",
                           dropout=0.1).to(DEV)
    ck = torch.load(CKPT[which], map_location=DEV)
    sd = ck["ema_model" if ema else "net_model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=True), None
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ------------------------------------------------------------------ statistics
def flip_stats(path=None):
    """(m1, Sigma1) of the FLIP-AUGMENTED training set.

    TorchCFM trains with RandomHorizontalFlip, so the model's target P1 is the
    horizontally symmetrised CIFAR-10, not the raw one.  v_G must use the moments
    of the distribution the network was actually trained toward.  (The FID
    reference stays the standard un-augmented train-50k -- different role.)
    """
    if path and os.path.exists(path):
        return torch.load(path)
    X = torch.tensor(load_cifar(True), device=DEV)
    Xf = X.reshape(-1, 3, 32, 32).flip(-1).reshape(-1, 3072)
    Xa = torch.cat([X, Xf]).double()
    m1 = Xa.mean(0)
    Xc = Xa - m1
    C = Xc.T @ Xc / (len(Xa) - 1)
    lam, U = torch.linalg.eigh(C)
    st = {"lam": lam.clamp_min(1e-8).float().cpu(), "U": U.float().cpu(),
          "m1": m1.float().cpu(),
          "profiles": {"vanilla": torch.zeros(3072)}, "L0": 0.0}
    if path:
        torch.save(st, path)
    return st
