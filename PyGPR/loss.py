import torch as tc
import numpy as np
from numpy import ndarray
from typing import Tuple
from .gpr import GPR

tc.set_default_tensor_type(tc.DoubleTensor)


class Loss():
    """
     Base class for cost functions for Gaussian process model
     selection
    """
    def __init__(self, model: GPR) -> None:
        self.model: GPR = model
        self.loss_value: float = NotImplemented
        self.grad_value: ndarray = NotImplemented
        return None

    def loss(self, params: ndarray) -> float:
        raise NotImplementedError

    def grad(self, params: ndarray) -> ndarray:
        raise NotImplementedError

    def loss_and_grad(self, params: ndarray) -> Tuple[float, ndarray]:
        raise NotImplementedError


class MLE(Loss):
    """
     Log Marginal Likelihood for hyperparameters.
    """
    def loss(self, params: ndarray) -> float:

        krn = self.model.cov.kernel(tc.from_numpy(params), self.model.x)
        krn.diagonal(dim1=-2, dim2=-1).add_(1e-7)
        krnchd = tc.cholesky(krn)

        y = self.model.y
        y = y.view(-1, y.shape[-1], 1)

        wt = tc.cholesky_solve(y, krnchd)

        wt.squeeze_(2)
        y.squeeze_(2)

        llhd = 0.5 * wt.mul_(y).sum(-1) + tc.log(
            tc.diagonal(krnchd, dim1=-2, dim2=-1)).sum(
                -1) + 0.5 * y.shape[-1] * tc.log(tc.tensor(2 * np.pi))

        llhd.squeeze_(0)

        self.loss_value = llhd.numpy()

        return llhd.numpy()

    def grad(self, params: ndarray) -> ndarray:

        krn, dkrn = self.model.cov.kernel_and_grad(tc.from_numpy(params),
                                                   self.model.x)
        krn.diagonal(dim1=-2, dim2=-1).add_(1e-7)
        krnchd = tc.cholesky(krn)

        y = self.model.y
        y = y.view(-1, y.shape[-1], 1)

        krnchd = krnchd.view(-1, krnchd.shape[-2], krnchd.shape[-1])
        dkrn = dkrn.view(-1, dkrn.shape[-3], dkrn.shape[-2], dkrn.shape[-1])

        wt = tc.cholesky_solve(y, krnchd)

        wt.squeeze_(2)
        y.squeeze_(2)

        kk = tc.cholesky_solve(dkrn, krnchd[:, None, :, :])

        wtb = wt[:, :, None].mul(wt[:, None, :])
        tr1 = dkrn.mul_(wtb[:, None, :, :]).sum((-1, -2))
        # tr1 = oes.contract('i,kij,j->k', wt, dkrn, wt, backend='torch')
        tr2 = tc.diagonal(kk, dim1=-1, dim2=-2).sum(-1)

        jac_llhd = tr1.sub_(tr2).mul_(-0.5)

        jac_llhd.squeeze_(0)

        self.grad_value = jac_llhd.numpy()

        return jac_llhd.numpy()

    def loss_and_grad(self, params: ndarray) -> Tuple[float, ndarray]:

        krn, dkrn = self.model.cov.kernel_and_grad(tc.from_numpy(params),
                                                   self.model.x)
        krn.diagonal(dim1=-2, dim2=-1).add_(1e-7)
        krnchd = tc.cholesky(krn)

        y = self.model.y
        y = y.view(-1, y.shape[-1], 1)

        wt = tc.cholesky_solve(y, krnchd)
        wt.squeeze_(2)
        y.squeeze_(2)
        wtb = wt[:, :, None].mul(wt[:, None, :])

        llhd = 0.5 * wt.mul_(y).sum(-1) + tc.log(
            tc.diagonal(krnchd, dim1=-2, dim2=-1)).sum(
                -1) + 0.5 * y.shape[-1] * tc.log(tc.tensor(2 * np.pi))

        llhd.squeeze_(0)

        krnchd = krnchd.view(-1, krnchd.shape[-2], krnchd.shape[-1])
        dkrn = dkrn.view(-1, dkrn.shape[-3], dkrn.shape[-2], dkrn.shape[-1])

        kk = tc.cholesky_solve(dkrn, krnchd[:, None, :, :])

        tr1 = dkrn.mul_(wtb[:, None, :, :]).sum((-1, -2))
        tr2 = tc.diagonal(kk, dim1=-1, dim2=-2).sum(-1)

        jac_llhd = tr1.sub_(tr2).mul_(-0.5)

        jac_llhd.squeeze_(0)

        self.loss_value = llhd.numpy()
        self.grad_value = jac_llhd.numpy()

        return (llhd.numpy(), jac_llhd.numpy())


# Docs hereafter

Loss.loss.__doc__ = """
Computes value of loss at the model parameters.

Parameters
----------
params: ndarray[nhp,]
   Parameters of the model at which the loss function is evaluated.

Returns
-------
float
    Value of the loss function.
"""

Loss.grad.__doc__ = """
Computes gradient of loss at the model parameters.

Parameters
----------
params: ndarray[nhp,]
   Parameter of the model at which the gradient of the loss function\
   is evaluated.

Returns
-------
ndarray[nhp,]
    Gradient of the loss function wrt model parameters.
"""

Loss.loss_and_grad.__doc__ = """
Computes loss function and its gradient at the model parameters.

Parameters
----------
params: ndarray[nhp,]
   Parameter of the model at which the loss function and its gradient\
   are evaluated.

Returns
-------
tuple(float, ndarray[nhp,])
    Loss function and its gradient wrt model parameters.
"""
