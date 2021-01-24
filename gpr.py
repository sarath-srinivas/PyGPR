import torch as tc
from torch import Tensor
from typing import Sequence
from .covar import Covar

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR:
    """
     Base class for Gaussian process regression models.
    """

    def __init__(self, x: Tensor, y: Tensor, cov: Covar) -> None:
        self.x: Tensor = x
        self.y: Tensor = y
        self.cov = cov

        self.params: Tensor = NotImplemented
        self.need_upd: bool = True

        return None

    def set_params(self, params: Tensor) -> None:
        self.params = tc.clone(params)
        self.need_upd = True
        return None

    def update(self) -> None:
        raise NotImplementedError

    def predict(self, xp: Tensor, var: str) -> Sequence[Tensor]:
        raise NotImplementedError

    def predict_var(self, xp: Tensor, **kwrgs: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_covar(self, xp: Tensor, **kwargs: Tensor) -> Tensor:
        raise NotImplementedError


class Exact_GP(GPR):
    """
     Exact GP model.
    """

    def __init__(self, x: Tensor, y: Tensor, cov: Covar) -> None:

        super().__init__(x, y, cov)

        self.params: Tensor = cov.init_params(x)

        self.krn: Tensor = NotImplemented
        self.wt: Tensor = NotImplemented
        self.krnchd: Tensor = NotImplemented

        self.need_upd: bool = True

        return None

    def update(self) -> None:
        if self.need_upd:
            self.krn = self.cov.kernel(self.params, self.x)
            self.krn.diagonal(dim1=-2, dim2=-1).add_(1e-7)
            self.krnchd = tc.cholesky(self.krn)
            self.wt = tc.cholesky_solve(
                self.y[..., None], self.krnchd
            ).squeeze_(-1)
            self.need_upd = False
        return None

    def predict(self, xp: Tensor, var: str = "full") -> Sequence[Tensor]:

        self.update()
        krns = self.cov.kernel(self.params, self.x, xp)
        ys = tc.bmm(
            krns.view(-1, *krns.shape[-2:]),
            self.wt.view(-1, self.wt.shape[-1], 1),
        )

        ys = ys.squeeze_()

        if var == "full":
            covars = self.predict_covar(xp, krns=krns)
        elif var == "diag":
            covars = self.predict_var(xp, krns=krns)
        else:
            covars = NotImplemented

        return [ys, covars]

    def predict_var(self, xp: Tensor, **kwargs: Tensor) -> Tensor:
        krns = kwargs["krns"]
        krnss = self.cov.kernel(self.params, xp)
        krnst = krns.transpose(-2, -1)
        lks = tc.cholesky_solve(krnst, self.krnchd)

        var = tc.diagonal(krnss, dim1=-2, dim2=-1).sub_(
            krns.mul_(lks.transpose(-2, -1)).sum(-1)
        )

        return var

    def predict_covar(self, xp: Tensor, **kwargs: Tensor) -> Tensor:
        krns = kwargs["krns"]
        krnss = self.cov.kernel(self.params, xp)
        krnst = krns.transpose(-2, -1)
        lks = tc.cholesky_solve(krnst, self.krnchd)

        covars = krnss.sub_(
            tc.bmm(
                krns.view(-1, *krns.shape[-2:]), lks.view(-1, *lks.shape[-2:])
            ).squeeze()
        )

        return covars


# Docs hereafter

GPR.predict.__doc__ = """
Get Gaussian process mean prediction with covariance for xp.

Parameters
----------
xp: Tensor[..., np, dim]
    Batched Evaluation samples at which the interpolation is required.
    It should be atleast 2D tensor.

var: string, optional
    If "full" computes the full prediction covariance
    if "diag" computes only diagonal of the prediction covariance.
    else do not computes covariance

Returns
-------
[ Tensor[..., np], Tensor[..., np, np] ]\
    or [ Tensor[..., np], Tensor[..., np] ]
    If var is "full", returns mean and covariance matrix.
    If var is "diag" , returns mean and variance.
    If var is anything else, returns mean and NotImplented
"""

GPR.predict_var.__doc__ = """
Variance estimate of the posterior Gaussian process at xp.
..:math: x^2 = y^2

Parameters
----------
xp: Tensor[..., np, dim]
    Batched Evaluation samples at which the interpolation is required.
    It should be atleast 2D tensor.

**kwargs: Tensor
    Keyword arguments.

Returns
-------
Tensor[np]
    The variance estimate of targets at xp.
"""

GPR.predict_covar.__doc__ = """
Covariance estimate of the posterior Gaussian process at xp.
..:math: x^2 = y^2

Parameters
----------
xp: Tensor[..., np, dim]
    Batched Evaluation samples at which the interpolation is required.
    It should be atleast 2D tensor.

**kwargs: Tensor
    Keyword arguments.

Returns
-------
Tensor[np, np]
    The covariance matrix of targets at xp.
"""
