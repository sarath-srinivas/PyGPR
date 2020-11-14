import torch as tc
from typing import Callable, Sequence

# from gpr import log_likelihood, jac_log_likelihood
from .covar import Squared_exponential, Covar, Compose
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

T_Covar = Callable[..., Covar]

n = (10, 100, 1000)
np = (5, 50, 500)
dim = (2, 5)

composes = ([Squared_exponential, Squared_exponential],
            [Squared_exponential, Squared_exponential, Squared_exponential])

tparams = list(product(composes, n, np, dim))


@pyt.mark.parametrize("covars, n, np, dim", tparams)
def test_compose_covar(covars: Sequence[T_Covar],
                       n: int,
                       np: int,
                       dim: int,
                       tol: float = 1e-7) -> None:
    x = tc.rand([n, dim])
    xp = tc.rand([np, dim])
    cov_c = Compose(x, covars)
    cov_c.params = tc.rand_like(cov_c.params)
    krn_c = cov_c.kernel(x, xp=xp)

    krn = tc.zeros_like(krn_c)

    chunks = [covar.params.shape[-1] for covar in cov_c.covars]
    params = cov_c.params.split(chunks, dim=-1)

    for i, covar in enumerate(covars):
        cov = covar(x)
        cov.params = params[i]
        krn.add_(cov.kernel(x, xp=xp))

    assert tc.allclose(krn_c, krn, atol=tol)

    return None


tparams = list(product(composes, n, dim))


@pyt.mark.parametrize("covars, n, dim", tparams)
def test_compose_deriv_covar(covars: Sequence[T_Covar],
                             n: int,
                             dim: int,
                             tol: float = 1e-7) -> None:
    x = tc.rand([n, dim])
    cov_c = Compose(x, covars)
    cov_c.params = tc.rand_like(cov_c.params)
    krn_c, dkrn_c = cov_c.kernel_and_grad(x)

    krn = tc.zeros_like(krn_c)

    chunks = [covar.params.shape[-1] for covar in cov_c.covars]
    params = cov_c.params.split(chunks, dim=-1)

    dkrns = []

    for i, covar in enumerate(covars):
        cov = covar(x)
        cov.params = params[i]
        krn.add_(cov.kernel_and_grad(x)[0])
        dkrns.append(cov.kernel_and_grad(x)[1])

    dkrn = tc.cat(dkrns, dim=-3)

    assert tc.allclose(krn_c, krn, atol=tol)
    assert tc.allclose(dkrn_c, dkrn, atol=tol)

    return None


def compose(x):
    return Compose(x, [Squared_exponential, Squared_exponential])


covars = (Squared_exponential, compose)

tparams = list(product(covars, n, dim))


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_symmetric(covar: T_Covar,
                         n: int,
                         dim: int,
                         tol: float = 1e-7) -> None:
    x = tc.rand([n, dim])
    cov = covar(x)
    cov.params = tc.rand_like(cov.params)
    krn = cov.kernel(x)

    assert tc.allclose(krn, krn.t(), atol=tol)

    return None


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_posdef(covar: T_Covar, n: int, dim: int, tol=1e-7) -> None:
    x = tc.rand([n, dim])
    cov = covar(x)
    cov.params = tc.rand_like(cov.params)
    krn = cov.kernel(x)

    eig = tc.eig(krn)[0][:, 0]

    assert tc.all(eig > 0)

    return None


@pyt.mark.parametrize("covar, n, dim", tparams)
def test_covar_batch(covar: T_Covar, n: int, dim: int) -> None:
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
def test_covar_deriv(covar: T_Covar,
                     n: int,
                     dim: int,
                     eps_diff: float = 1e-5) -> None:
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
