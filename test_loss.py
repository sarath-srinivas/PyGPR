import torch as tc
import numpy as np
from .gpr import Exact_GP
from .covar import Squared_exponential
from .loss import MLE
from itertools import product
import pytest as pyt

tc.set_default_tensor_type(tc.DoubleTensor)

n = (10, 100, 1000)
dim = (2, 3, 5, 7)

tparams = list(product(n, dim))


@pyt.mark.parametrize("n, dim", tparams)
def test_grad(n: int, dim: int, eps_diff: float = 1e-5) -> None:
    x = tc.rand([n, dim])
    y = tc.exp(-x.square().sum(1))

    cov = Squared_exponential()

    mod = Exact_GP(x, y, cov)

    loss = MLE(mod)

    params = tc.rand_like(mod.params).numpy()

    grad = loss.grad(params)

    grad_diff = np.zeros_like(grad)

    for k in range(0, params.shape[-1]):
        eps = np.zeros_like(params)
        eps[k] = eps_diff

        val = loss.loss(params)
        params_eps = params + eps
        val_eps = loss.loss(params_eps)

        grad_diff[k] = (val_eps - val) / eps_diff

    assert np.allclose(np.log10(np.abs(grad)),
                       np.log10(np.abs(grad_diff)),
                       atol=1e-3)
