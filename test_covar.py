import torch as tc
from typing import Any

# from gpr import log_likelihood, jac_log_likelihood
from .covar import Squared_exponential, Covar
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

n = (10, 100, 1000)
dim = (2, 5)

covars = (Squared_exponential,)

tparams = list(product(covars, n, dim))


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_symmetric(covar: Any, n: int, dim: int, tol: float = 1e-7) \
        -> None:
    x = tc.rand([n, dim])
    cov = covar(x)
    cov.params = tc.rand_like(cov.params)
    krn = cov.kernel(x)

    assert tc.allclose(krn, krn.t(), atol=tol)

    return None


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_posdef(covar: Any, n: int, dim: int, tol=1e-7) -> None:
    x = tc.rand([n, dim])
    cov = covar(x)
    cov.params = tc.rand_like(cov.params)
    krn = cov.kernel(x)

    eig = tc.eig(krn)[0][:, 0]

    assert tc.all(eig > 0)

    return None


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_batch(covar: Any, n: int, dim: int) -> None:
    nc = 4
    xb = tc.rand(nc, n, dim)
    covb = covar(xb)
    covb.params = tc.rand_like(covb.params)

    krn_batch, dkrn_batch = covb.kernel_and_grad(xb)

    cov = covar(xb[0, :, :])

    cov.params = covb.params[0, :]
    krn1, dkrn1 = cov.kernel_and_grad(xb[0, :, :])
    cov.params = covb.params[1, :]
    krn2, dkrn2 = cov.kernel_and_grad(xb[1, :, :])
    cov.params = covb.params[2, :]
    krn3, dkrn3 = cov.kernel_and_grad(xb[2, :, :])
    cov.params = covb.params[3, :]
    krn4, dkrn4 = cov.kernel_and_grad(xb[3, :, :])

    krn = tc.stack((krn1, krn2, krn3, krn4), dim=0)
    dkrn = tc.stack((dkrn1, dkrn2, dkrn3, dkrn4), dim=0)

    assert tc.allclose(krn_batch, krn)
    assert tc.allclose(dkrn_batch, dkrn)

    return None


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_deriv(covar: Any, n: int, dim: int, eps_diff: float = 1e-5) \
        -> None:
    x = tc.rand(n, dim)
    cov: Covar = covar(x)
    hp = tc.rand_like(cov.params)
    nhp = cov.params.shape[-1]

    cov.params = tc.clone(hp)
    krn, dkrn = cov.kernel_and_grad(x)

    dkrn_diff = tc.ones_like(dkrn)

    for k in range(0, nhp):
        eps = tc.zeros_like(hp)
        eps[k] = eps_diff

        cov.params = tc.clone(hp)
        krn = cov.kernel(x)
        cov.params.add_(eps)
        krn_eps = cov.kernel(x)

        dkrn_diff[k, :, :] = krn_eps.sub_(krn).div_(eps_diff)

    assert tc.allclose(dkrn, dkrn_diff, atol=eps_diff)

    return None


"""
@pyt.mark.parametrize("n,covar_fun", tparams)
def test_jac_likelihood(n, covar_fun, eps_diff=1e-5):
    dim = 5
    cov = covars[covar_fun]
    x = tc.rand(n, dim)
    y = tc.exp(-x.square().sum(1))
    nhp = cov(x).shape[-1]
    hp = tc.ones(nhp)

    jac_llhd = tc.tensor(jac_log_likelihood(x, y, hp, cov))

    jac_llhd_diff = tc.zeros_like(jac_llhd)

    for k in range(0, nhp):
        eps = tc.zeros(nhp)
        eps[k] = eps_diff
        hp_eps = hp.add(eps)

        llhd = tc.tensor(log_likelihood(x, y, hp, cov))
        llhd_eps = tc.tensor(log_likelihood(x, y, hp_eps, cov))

        jac_llhd_diff[k] = llhd_eps.sub_(llhd).div_(eps_diff)

    assert tc.allclose(jac_llhd, jac_llhd_diff, atol=1e-3)
"""
