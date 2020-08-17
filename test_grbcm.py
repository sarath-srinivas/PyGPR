import torch as tc
import numpy as np
from .gr_bcm import GRBCM
from .covar import covars
from itertools import product
import pytest as pyt

dim = (2, 3, 7)
n = (10, 50, 100)
nc = (2, 5, 10)
ng = (10, 100)

tparams = list(product(ng, nc, n, dim, covars))


@pyt.mark.parametrize("ng, nc,n,dim,covar_fun", tparams)
def test_interpolate(ng, nc, n, dim, covar_fun):
    xl = tc.rand(nc, n, dim)
    xg = tc.rand(ng, dim)
    yl = tc.sin(xl.sum(-1))
    yg = tc.sin(xg.sum(-1))

    cov = covars[covar_fun]

    idx = tc.randperm(xl.shape[0])[0]

    xs = tc.clone(xl[idx, :, :])
    ys = tc.sin(xs.sum(-1))

    gp = GRBCM(xl, yl, xg, yg, cov)

    ys_gpr, covar_s = gp.interpolate(xs, diag_only=True)

    assert tc.allclose(ys, ys_gpr, atol=1e-4)
    assert tc.all(tc.diag(covar_s) < 1e6)
