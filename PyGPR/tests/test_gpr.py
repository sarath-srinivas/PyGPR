import torch as tc
from PyGPR import Exact_GP
from PyGPR import Covar, Squared_exponential, White_noise, Compose
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

dim = (2, 3, 7)
n = (10, 50, 100)

covars = ([Squared_exponential(), White_noise()],)

tparams = list(product(n, dim, covars))


@pyt.mark.parametrize("n,dim,covars", tparams)
def test_interpolate(n: int, dim: int, covars: Covar) -> None:
    x = tc.rand(n, dim)
    y = tc.sin(-x.sum(-1))

    cov = Compose(covars)

    xs = tc.clone(x)

    gp = Exact_GP(x, y, cov)

    ys, covar_s = gp.predict(xs, var="diag")

    assert tc.allclose(ys, y, atol=1e-4)
    assert covar_s.shape == ys.shape
    assert tc.all(tc.diag(covar_s) < 1e6)


@pyt.mark.parametrize("n,dim,covars", tparams)
def test_pred_covar(
    n: int, dim: int, covars: Covar, tol: float = 1e-7
) -> None:
    x = tc.rand(n, dim)
    y = tc.sin(-x.sum(-1))

    cov = Compose(covars)

    xs = tc.clone(x)

    gp = Exact_GP(x, y, cov)

    ys, covar_s = gp.predict(xs)
    eig = tc.eig(covar_s)[0][:, 0]

    assert tc.allclose(covar_s, covar_s.t(), atol=tol)
    assert tc.all(eig > -tol)


nc = (2, 5, 10)
tparams = list(product(nc, n, dim, covars))


@pyt.mark.parametrize("nc,n,dim,covars", tparams)
def test_interpolate_batch(nc: int, n: int, dim: int, covars: Covar) -> None:
    xl = tc.rand(n, dim)
    x = tc.empty(nc, n, dim).copy_(xl)
    y = tc.sin(-x.sum(-1))

    cov = Compose(covars)

    xs = tc.clone(xl)
    ys = tc.sin(-xl.sum(-1))

    gp = Exact_GP(x, y, cov)

    ys, covar_s = gp.predict(xs, var="diag")

    assert ys.shape == (nc, xs.shape[0])
    assert tc.allclose(ys, y.reshape(nc, -1), atol=1e-4)
    assert tc.all(tc.diag(covar_s) < 1e6)


@pyt.mark.parametrize("nc,n,dim,covars", tparams)
def test_pred_covar_batch(
    nc: int, n: int, dim: int, covars: Covar, tol: float = 1e-7
) -> None:
    xl = tc.rand(n, dim)
    x = tc.empty(nc, n, dim).copy_(xl)
    y = tc.sin(-x.sum(-1))

    cov = Compose(covars)

    xs = tc.clone(xl)
    ys = tc.sin(-xl.sum(-1))

    gp = Exact_GP(x, y, cov)

    ys, covar_s = gp.predict(xs)
    eig = tc.symeig(covar_s)[0]

    assert covar_s.shape == (nc, n, n)
    assert tc.all(tc.diagonal(covar_s, dim1=-2, dim2=-1) < 1e-6)
    assert tc.allclose(covar_s, covar_s.transpose(-2, -1), atol=tol)
    assert tc.all(eig > -tol)
