import torch as tc
import numpy as np
import scipy as scp
from .gpr import log_likelihood, jac_log_likelihood
from .covar import covars
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

n = (10, 100, 1000)

tparams = list(product(n, covars))


@pyt.mark.parametrize("n,covar_fun", tparams)
def test_covar_symmetric(n, covar_fun, tol=1e-7):
    dim = 5
    cov = covars[covar_fun]
    x = tc.rand([n, dim])
    hp = tc.rand(cov(x).shape[0])
    krn = cov(x, hp=hp)

    assert tc.allclose(krn, krn.t(), atol=tol)


@pyt.mark.parametrize("n,covar_fun", tparams)
def test_covar_posdef(n, covar_fun, tol=1e-7):
    dim = 5
    cov = covars[covar_fun]
    x = tc.rand([n, dim])
    hp = tc.rand(cov(x).shape[0])
    krn = cov(x, hp=hp)
    eig = tc.eig(krn)[0][:, 0]

    assert tc.all(eig > 0)


@pyt.mark.parametrize("n,covar_fun", tparams)
def test_covar_batch(n, covar_fun):
    nc = 4
    dim = 5
    cov = covars[covar_fun]
    xb = tc.rand(nc, n, dim)
    nhp = cov(xb[0, :, :]).shape[-1]
    hpb = tc.rand(nc, nhp)

    krn_batch, dkrn_batch = cov(xb, hp=hpb, deriv=True)

    krn1, dkrn1 = cov(xb[0, :, :], hp=hpb[0, :], deriv=True)
    krn2, dkrn2 = cov(xb[1, :, :], hp=hpb[1, :], deriv=True)
    krn3, dkrn3 = cov(xb[2, :, :], hp=hpb[2, :], deriv=True)
    krn4, dkrn4 = cov(xb[3, :, :], hp=hpb[3, :], deriv=True)

    krn = tc.stack((krn1, krn2, krn3, krn4), dim=0)
    dkrn = tc.stack((dkrn1, dkrn2, dkrn3, dkrn4), dim=0)

    assert tc.allclose(krn_batch, krn)
    assert tc.allclose(dkrn_batch, dkrn)


@pyt.mark.parametrize("n,covar_fun", tparams)
def test_covar_deriv(n, covar_fun, eps_diff=1e-5):
    dim = 5
    cov = covars[covar_fun]
    x = tc.rand(n, dim)
    nhp = cov(x).shape[-1]
    hp = tc.rand(nhp)

    krn, dkrn = cov(x, hp=hp, deriv=True)

    dkrn_diff = tc.ones_like(dkrn)

    for k in range(0, nhp):
        eps = tc.zeros(nhp)
        eps[k] = eps_diff
        hp_eps = hp.add(eps)

        krn = cov(x, hp=hp)
        krn_eps = cov(x, hp=hp_eps)

        dkrn_diff[k, :, :] = krn_eps.sub_(krn).div_(eps_diff)

    #return dkrn, dkrn_diff
    assert tc.allclose(dkrn, dkrn_diff, atol=eps_diff)


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
