import torch as tc
from typing import Sequence

# from gpr import log_likelihood, jac_log_likelihood
from PyGPR import Squared_exponential, Covar, Compose, White_noise
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

n = (10, 100, 1000)
np = (5, 50, 500)
dim = (2, 5)

composes = (
    [Squared_exponential(), Squared_exponential()],
    [Squared_exponential(), White_noise()],
    [Squared_exponential(), Squared_exponential(), White_noise()],
)

tparams = list(product(composes, n, np, dim))


@pyt.mark.parametrize("covars, n, np, dim", tparams)
def test_compose_covar(
    covars: Sequence[Covar], n: int, np: int, dim: int, tol: float = 1e-7
) -> None:
    x = tc.rand([n, dim])
    xp = tc.rand([np, dim])
    cov_c = Compose(covars)
    hp_shape = cov_c.get_params_shape(x)
    hp = tc.rand(hp_shape)
    krn_c = cov_c.kernel(hp, x, xp=xp)

    krn = tc.zeros_like(krn_c)

    chunks = [covar.get_params_shape(x)[-1] for covar in cov_c.covars]
    params = hp.split(chunks, dim=-1)

    for i, cov in enumerate(covars):
        krn.add_(cov.kernel(params[i], x, xp=xp))

    assert tc.allclose(krn_c, krn, atol=tol)

    return None


tparams1 = list(product(composes, n, dim))


@pyt.mark.parametrize("covars, n, dim", tparams1)
def test_compose_deriv_covar(
    covars: Sequence[Covar], n: int, dim: int, tol: float = 1e-7
) -> None:
    x = tc.rand([n, dim])
    cov_c = Compose(covars)
    hp_shape = cov_c.get_params_shape(x)
    hp = tc.rand(hp_shape)
    krn_c, dkrn_c = cov_c.kernel_and_grad(hp, x)

    krn = tc.zeros_like(krn_c)

    chunks = [covar.get_params_shape(x)[-1] for covar in cov_c.covars]
    params = hp.split(chunks, dim=-1)

    dkrns = []

    for i, cov in enumerate(covars):
        krn.add_(cov.kernel_and_grad(params[i], x)[0])
        dkrns.append(cov.kernel_and_grad(params[i], x)[1])

    dkrn = tc.cat(dkrns, dim=-3)

    assert tc.allclose(krn_c, krn, atol=tol)
    assert tc.allclose(dkrn_c, dkrn, atol=tol)

    return None


compose = Compose(
    [Squared_exponential(), Squared_exponential(), White_noise()]
)


covars = (Squared_exponential(), White_noise(), compose)

tparams2 = list(product(covars, n, dim))


@pyt.mark.parametrize("cov, n, dim", tparams2)
def test_covar_symmetric(
    cov: Covar, n: int, dim: int, tol: float = 1e-7
) -> None:
    x = tc.rand([n, dim])
    hp = tc.rand(cov.get_params_shape(x))
    krn = cov.kernel(hp, x)

    assert tc.allclose(krn, krn.t(), atol=tol)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams2)
def test_covar_posdef(cov: Covar, n: int, dim: int, tol=1e-7) -> None:
    x = tc.rand([n, dim])
    hp = tc.rand(cov.get_params_shape(x))
    krn = cov.kernel(hp, x)

    krn.add_(1e-7 * tc.eye(n))
    eig = tc.linalg.eigvals(krn)

    assert tc.all(tc.real(eig) > 0)
    assert tc.all(tc.imag(eig).abs_() < 1e-7)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams2)
def test_covar_batch(cov: Covar, n: int, dim: int) -> None:
    nc = 4
    xb = tc.rand(nc, n, dim)
    hpb = tc.rand(cov.get_params_shape(xb))

    krn_batch, dkrn_batch = cov.kernel_and_grad(hpb, xb)

    krn1, dkrn1 = cov.kernel_and_grad(hpb[0, :], xb[0, :, :])
    krn2, dkrn2 = cov.kernel_and_grad(hpb[1, :], xb[1, :, :])
    krn3, dkrn3 = cov.kernel_and_grad(hpb[2, :], xb[2, :, :])
    krn4, dkrn4 = cov.kernel_and_grad(hpb[3, :], xb[3, :, :])

    krn = tc.stack((krn1, krn2, krn3, krn4), dim=0)
    dkrn = tc.stack((dkrn1, dkrn2, dkrn3, dkrn4), dim=0)

    assert tc.allclose(krn_batch, krn)
    assert tc.allclose(dkrn_batch, dkrn)

    return None


@pyt.mark.parametrize("cov, n, dim", tparams2)
def test_covar_deriv(
    cov: Covar, n: int, dim: int, eps_diff: float = 1e-5
) -> None:
    x = tc.rand(n, dim)
    hp = tc.rand(cov.get_params_shape(x))
    nhp = hp.shape[-1]

    krn, dkrn = cov.kernel_and_grad(hp, x)

    dkrn_diff = tc.ones_like(dkrn)

    for k in range(0, nhp):
        eps = tc.zeros_like(hp)
        eps[k] = eps_diff

        krn = cov.kernel(hp, x)
        krn_eps = cov.kernel(hp.add(eps), x)

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
