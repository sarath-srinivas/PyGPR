import torch as tc
import numpy as np
from .gpr import GPR
from .covar import covars
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

dim = (2, 3, 7)
n = (10, 50, 100)

tparams = list(product(n, dim, covars))


@pyt.mark.parametrize("n,dim,covar_fun", tparams)
def test_interpolate(n, dim, covar_fun):
    x = tc.rand(n, dim)
    y = tc.sin(-x.sum(-1))

    cov = covars[covar_fun]

    xs = tc.clone(x)

    gp = GPR(x, y, cov)

    ys, covar_s = gp.interpolate(xs, diag_only=True)

    assert tc.allclose(ys, y, atol=1e-4)
    assert covar_s.shape == ys.shape
    assert tc.all(tc.diag(covar_s) < 1e6)


@pyt.mark.parametrize("n,dim,covar_fun", tparams)
def test_pred_covar(n, dim, covar_fun, tol=1e-7):
    x = tc.rand(n, dim)
    y = tc.sin(-x.sum(-1))

    cov = covars[covar_fun]

    xs = tc.clone(x)

    gp = GPR(x, y, cov)

    ys, covar_s = gp.interpolate(xs)
    eig = tc.eig(covar_s)[0][:, 0]

    assert tc.allclose(covar_s, covar_s.t(), atol=tol)
    assert tc.all(eig > 0)


nc = (2, 5, 10)
tparams = list(product(nc, n, dim, covars))


@pyt.mark.parametrize("nc,n,dim,covar_fun", tparams)
def test_interpolate_batch(nc, n, dim, covar_fun):
    xl = tc.rand(n, dim)
    x = tc.empty(nc, n, dim).copy_(xl)
    y = tc.sin(-x.sum(-1))

    cov = covars[covar_fun]

    xs = tc.clone(xl)
    ys = tc.sin(-xl.sum(-1))

    gp = GPR(x, y, cov)

    ys, covar_s = gp.interpolate(xs, diag_only=True)

    assert ys.shape == (nc, xs.shape[0])
    assert tc.allclose(ys, y.reshape(nc, -1), atol=1e-4)
    assert tc.all(tc.diag(covar_s) < 1e6)


@pyt.mark.parametrize("nc,n,dim,covar_fun", tparams)
def test_pred_covar_batch(nc, n, dim, covar_fun, tol=1e-7):
    xl = tc.rand(n, dim)
    x = tc.empty(nc, n, dim).copy_(xl)
    y = tc.sin(-x.sum(-1))

    cov = covars[covar_fun]

    xs = tc.clone(xl)
    ys = tc.sin(-xl.sum(-1))

    gp = GPR(x, y, cov)

    ys, covar_s = gp.interpolate(xs)
    eig = tc.symeig(covar_s)[0]

    assert covar_s.shape == (nc, n, n)
    assert tc.all(tc.diagonal(covar_s, dim1=-2, dim2=-1) < 1e-6)
    assert tc.allclose(covar_s, covar_s.transpose(-2, -1), atol=tol)
    assert tc.all(eig > 0)
