import torch as tc
from torch import Tensor
from .gpr import GPR

tc.set_default_tensor_type(tc.DoubleTensor)


class Var():
    """
     Base class interface for prediction covariance estimates
     for Gaussian process.
    """

    def __init__(self) -> None:
        return None

    def pred_covar(self, xp: Tensor, gp: GPR, **kwargs: Tensor) -> Tensor:
        raise NotImplementedError

    def pred_var(self, xp: Tensor, gp: GPR, **kwargs: Tensor) -> Tensor:
        raise NotImplementedError


class Exact_var(Var):
    """
     Computes prediction covariance exactly using
     V(xp, xp) = K(xp, xp) - K(xp, x) K(x, x)^-1 K(x, xp)
    """

    def pred_covar(self, xp: Tensor, gp: GPR, **kwargs: Tensor) -> Tensor:
        krns = kwargs["krns"]
        krnss = gp.cov.kernel(xp)
        krnst = krns.transpose(-2, -1)
        lks = tc.cholesky_solve(krnst, gp.krnchd)

        covars = krnss.sub_(tc.bmm(krns.view(-1, *krns.shape[-2:]),
                            lks.view(-1, *lks.shape[-2:])).squeeze())

        return covars

    def pred_var(self, xp: Tensor, gp: GPR, **kwargs: Tensor) -> Tensor:
        krns = kwargs["krns"]
        krnss = gp.cov.kernel(xp)
        krnst = krns.transpose(-2, -1)
        lks = tc.cholesky_solve(krnst, gp.krnchd)

        var = tc.diagonal(krnss, dim1=-2, dim2=-1).sub_(
                            krns.mul_(lks.transpose(-2, -1)).sum(-1))

        return var
