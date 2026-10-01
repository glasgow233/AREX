"""Moment-matched Gaussian surrogate and the exact affine propagator (Sec. 2 of the paper).

In the eigenbasis U of the target covariance Sigma_1 = U diag(lam) U^T the affine flow decouples into one
scalar mode per direction: sigma_j(t)^2 = (1-t)^2 + t^2 lam_j, k_j(t) = d/dt log sigma_j(t), and the
propagator F(t, s) = diag(sigma_j(t) / sigma_j(s)).  `w_kernel` integrates F(t, u) against a weight,
which gives the residual quadrature weights of the one-node and Adams--Bashforth rules.
"""
import torch
from arex.common import DEV, TMIN, load_cifar
def make_ts(grid, steps):
    """Time grids. 'shift3' (SD3-style shifting) densifies steps near t=1;
    'pow2r' t = 1-(1-u)^2 likewise; 'unif' is what all main results used."""
    u = torch.linspace(0, 1, steps + 1)
    if grid == "unif":
        s = u
    elif grid == "shift3":
        s = 3 * u / (1 + 2 * u)
    elif grid == "pow2r":
        s = 1 - (1 - u) ** 2
    else:
        raise ValueError(grid)
    return (s * (1 - TMIN)).tolist()


class Surrogate:
    """Diagonal-in-some-basis Gaussian surrogate + exact propagator machinery."""

    def __init__(self, kind, st, moments_n=None, ridge=0.0, rank=None, sval="mean"):
        self.kind = kind
        full_lam = st["lam"].to(DEV)
        full_U = st["U"].to(DEV)
        m1_pix = st["m1"].to(DEV)

        if moments_n is not None:  # re-estimate moments from a small subset
            X = torch.tensor(load_cifar(True), device=DEV)[:moments_n]
            m1_pix = X.mean(0)
            Xc = (X - m1_pix).double()
            C = Xc.T @ Xc / max(len(X) - 1, 1)
            if ridge > 0:
                C += ridge * float(C.diagonal().mean()) * torch.eye(C.shape[0], device=DEV, dtype=torch.float64)
            lam64, U64 = torch.linalg.eigh(C)
            full_lam = lam64.clamp_min(1e-8).float()
            full_U = U64.float()

        if kind in ("eigen",):
            self.U, lam = full_U, full_lam.clone()
            if rank is not None:  # keep top-`rank` eigenvalues, isotropic tail
                d = lam.numel()
                tail = lam[: d - rank].mean()
                lam[: d - rank] = tail
            self.lam = lam
        elif kind == "pixel":
            # diagonal of Sigma1 in pixel space
            X = torch.tensor(load_cifar(True), device=DEV)
            m1_pix = X.mean(0)
            self.U = None
            self.lam = ((X - m1_pix) ** 2).mean(0).clamp_min(1e-8)
        elif kind == "scalar":
            self.U = None
            v = {"mean": float(full_lam.mean()),
                 "geo": float(full_lam.log().mean().exp()),
                 "one": 1.0}[sval]
            self.lam = torch.full_like(full_lam, v)
        else:
            raise ValueError(kind)
        self.m1e = m1_pix if self.U is None else m1_pix @ self.U
        self.d = self.lam.numel()

    def to_basis(self, x):
        return x if self.U is None else x @ self.U

    def from_basis(self, xe):
        return xe if self.U is None else xe @ self.U.T

    def sig(self, t):
        return torch.sqrt((1 - t) ** 2 + t ** 2 * self.lam)

    def k(self, t):
        return (-(1 - t) + t * self.lam) / self.sig(t) ** 2

    def v_G(self, t, xe):
        return self.m1e + self.k(t) * (xe - t * self.m1e)

    def prop(self, s, t, xe):
        return t * self.m1e + (self.sig(t) / self.sig(s)) * (xe - s * self.m1e)

    def w_kernel(self, s, t, weight, nq=160):
        """int_s^t sigma_t/sigma_u * weight(u) du, per direction."""
        u = torch.linspace(s, t, nq, device=DEV, dtype=torch.float64)[:, None]
        lam = self.lam.double()[None, :]
        sig_u = torch.sqrt((1 - u) ** 2 + u ** 2 * lam)
        sig_t = torch.sqrt(torch.tensor((1 - t) ** 2, device=DEV, dtype=torch.float64) + t ** 2 * lam)
        wgt = weight(u) if weight is not None else 1.0
        return torch.trapezoid(sig_t / sig_u * wgt, u.squeeze(1), dim=0).float()

