import torch as tc
from lib_gpr.opt import Opt, CG_Quad, BFGS_Quad
from lib_gpr.loss import Loss
import pytest as pyt
from itertools import product
from typing import Callable
import numpy as np

tc.set_default_tensor_type(tc.DoubleTensor)

dim = (2, 3, 5, 7)
seed = (23, 443556, 1233)
opts = (CG_Quad, BFGS_Quad)

tparams = list(product(dim, opts, seed))

T_OPT = Callable[..., Opt]


@pyt.mark.parametrize("dim, optim, seed", tparams)
def test_opt_quad(dim: int, optim: T_OPT, seed: int) -> None:
    def grad(x, H, J):
        return J + np.matmul(H, x)

    def fun(x, H, J):
        return np.dot(J, x) + np.dot(x, np.matmul(H, x))

    np.random.seed(seed)

    L = np.random.rand(dim, dim)
    H = np.matmul(L.T, L)

    eig_H = np.linalg.eigvals(H)

    kappa = np.max(eig_H) / np.min(eig_H)

    assert np.alltrue(eig_H)

    J = np.random.rand(dim)

    loss = Loss(None)
    loss.loss = lambda x: fun(x, H, J)
    loss.grad = lambda x: grad(x, H, J)

    xmin_exact = np.linalg.solve(H, -J)

    par = np.random.rand(dim)

    opt = optim(loss, par)

    k = opt.minimize()

    print("iter:", k)
    print("kappa:", kappa)

    assert np.allclose(opt.x, xmin_exact, rtol=1e-3)
