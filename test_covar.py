import torch as tc

# from gpr import log_likelihood, jac_log_likelihood
from .covar import Squared_exponential, Covar
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

n = (10, 100, 1000)
dim = (2, 5)

sq_exp = Squared_exponential()

covars = (sq_exp,)

tparams = list(product(covars, n, dim))


@pyt.mark.parametrize("cov, n, dim", tparams)
def test_covar_symmetric(cov: Covar, n: int, dim: int, tol: float = 1e-7) \
        -> None:
    x = tc.rand([n, dim])
    hp = tc.rand([cov.hyper_param_size(x)])
    krn = cov.kernel(x, hp=hp)

    assert tc.allclose(krn, krn.t(), atol=tol)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams)
def test_covar_posdef(cov: Covar, n: int, dim: int, tol=1e-7) -> None:
    x = tc.rand([n, dim])
    hp = tc.rand([cov.hyper_param_size(x)])
    krn = cov.kernel(x, hp=hp)
    eig = tc.eig(krn)[0][:, 0]

    assert tc.all(eig > 0)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams)
def test_covar_batch(cov: Covar, n: int, dim: int) -> None:
    nc = 4
    xb = tc.rand(nc, n, dim)
    nhp = cov.hyper_param_size(xb)
    hpb = tc.rand(nc, nhp)

    krn_batch, dkrn_batch = cov.kernel_and_grad(xb, hp=hpb)

    krn1, dkrn1 = cov.kernel_and_grad(xb[0, :, :], hp=hpb[0, :])
    krn2, dkrn2 = cov.kernel_and_grad(xb[1, :, :], hp=hpb[1, :])
    krn3, dkrn3 = cov.kernel_and_grad(xb[2, :, :], hp=hpb[2, :])
    krn4, dkrn4 = cov.kernel_and_grad(xb[3, :, :], hp=hpb[3, :])

    krn = tc.stack((krn1, krn2, krn3, krn4), dim=0)
    dkrn = tc.stack((dkrn1, dkrn2, dkrn3, dkrn4), dim=0)

    assert tc.allclose(krn_batch, krn)
    assert tc.allclose(dkrn_batch, dkrn)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams)
def test_covar_deriv(cov: Covar, n: int, dim: int, eps_diff: float = 1e-5) \
        -> None:
    x = tc.rand(n, dim)
    nhp = cov.hyper_param_size(x)
    hp = tc.rand(nhp)

    krn, dkrn = cov.kernel_and_grad(x, hp=hp)

    dkrn_diff = tc.ones_like(dkrn)

    for k in range(0, nhp):
        eps = tc.zeros(nhp)
        eps[k] = eps_diff
        hp_eps = hp.add(eps)

        krn = cov.kernel(x, hp=hp)
        krn_eps = cov.kernel(x, hp=hp_eps)

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
