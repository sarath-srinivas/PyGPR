import torch as tc
import numpy as np
from numpy import ndarray
from .loss import Loss
import scipy.optimize as scopt
from typing import Callable

tc.set_default_tensor_type(tc.DoubleTensor)


class Opt:
    """
     Base class for optimisers.
    """

    def __init__(self, loss: Loss, par: ndarray = None) -> None:
        self.loss: Loss = loss
        self.args: dict = {}
        self.x: ndarray = NotImplemented
        return None

    def minimize(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class CG(Opt):
    """
     Conjugate gradient optimizer
    """

    def __init__(self, loss: Loss) -> None:
        super().__init__(loss)

        self.args = {
            "gtol": 1e-4,
            "maxiter": 1000,
            "disp": True,
            "return_all": True,
        }
        self.res: scopt.OptimizeResult = NotImplemented

    def minimize(self) -> None:
        params = tc.clone(self.loss.model.params).numpy()

        self.fstr = open("opt.dat", "w")

        self.res = scopt.minimize(
            self.loss.loss_and_grad,
            params,
            method="CG",
            jac=True,
            callback=self.callback,
            options=self.args,
        )

        self.fstr.close()

        if self.res.success is True:
            self.loss.model.set_params(tc.from_numpy(self.res.x))
        else:
            self.loss.model.set_params(tc.from_numpy(self.res.x))
            print("Optimizer Failed")

        return None

    def callback(self, params: ndarray) -> None:
        print(
            *params,
            self.loss.loss_value,
            np.linalg.norm(self.loss.grad_value),
            file=self.fstr,
        )

    def step(self):
        raise NotImplementedError


class Nelder_Mead(Opt):
    """
     Nelder Mead optimizer
    """

    def __init__(self, loss: Loss) -> None:
        super().__init__(loss)

        self.args = {
            "fatol": 1e-4,
            "maxiter": 1000,
            "disp": True,
            "return_all": True,
        }

        self.res: scopt.OptimizeResult = NotImplemented

    def minimize(self) -> None:
        params = tc.clone(self.loss.model.params).numpy()

        self.fstr = open("opt.dat", "w")
        self.res = scopt.minimize(
            self.loss.loss,
            params,
            method="Nelder-Mead",
            callback=self.callback,
            options=self.args,
        )
        self.fstr.close()

        if self.res.success is True:
            self.loss.model.set_params(tc.from_numpy(self.res.x))
        else:
            print("Optimizer Failed")

        return None

    def callback(self, params: ndarray) -> None:
        print(*params, self.loss.loss_value, file=self.fstr)

    def step(self):
        raise NotImplementedError


def hessian(
    x: np.ndarray, jac: Callable[..., np.ndarray], eps: float
) -> np.ndarray:
    dim = x.shape[-1]
    hess = np.empty([dim, dim])

    for i in range(0, dim):
        x_eps = np.copy(x)
        x_eps[i] += eps
        hess[:, i] = (jac(x_eps) - jac(x)) / eps

    return hess


class CG_Quad(Opt):
    """
     Linear Conjugate Gradient algorithm
    """

    def __init__(
        self,
        loss: Loss,
        par: ndarray = None,
        gtol: float = 1e-4,
        max_iter: int = 100,
        fd_eps: float = 1e-5,
    ) -> None:

        super().__init__(loss)
        self.x: ndarray = loss.model.params.numpy() if par is None else par
        self.r: ndarray = loss.grad(par)
        self.p: ndarray = -1.0 * self.r
        self.eps = fd_eps
        self.max_iter = max_iter
        self.gtol = gtol
        return None

    def hessian_product(self, par: ndarray, v: ndarray, eps: float) -> ndarray:
        Hv = (self.loss.grad(par + eps * v) - self.loss.grad(par)) / eps
        return Hv

    def step(self) -> None:
        r = np.copy(self.r)
        p = np.copy(self.p)
        x = np.copy(self.x)

        Hp = self.hessian_product(x, p, eps=self.eps)

        rr = np.dot(r, r)

        alp = rr / np.dot(p, Hp)

        x = x + alp * p
        r = r + alp * Hp
        bet = np.dot(r, r) / rr
        p = bet * p - r

        self.r = np.copy(r)
        self.p = np.copy(p)
        self.x = np.copy(x)

        return None

    def minimize(self) -> int:

        k = 0
        gnorm = np.linalg.norm(self.r)

        fstr = open("opt.dat", "w")
        while gnorm > self.gtol and k < self.max_iter:
            self.step()
            gnorm = np.linalg.norm(self.r)
            k = k + 1
            print(k, gnorm, file=fstr)

        fstr.close()

        if self.loss.model is not None:
            self.loss.model.set_params(tc.tensor(self.x))

        return k


class BFGS_Quad(Opt):
    """
     BFGS fror quadratic function.
    """

    def __init__(
        self,
        loss: Loss,
        par: ndarray = None,
        H0: ndarray = None,
        gtol: float = 1e-4,
        max_iter: int = 100,
        fd_eps: float = 1e-5,
    ) -> None:

        super().__init__(loss)
        self.x: ndarray = loss.model.params.numpy() if par is None else par
        self.r: ndarray = loss.grad(par)
        self.HI: ndarray = np.identity(
            self.x.shape[-1]
        ) if H0 is None else np.linalg.inv(H0)
        self.s: ndarray = NotImplemented
        self.y: ndarray = NotImplemented
        self.eps = fd_eps
        self.gtol = gtol
        self.max_iter = max_iter
        return None

    def hessian_inv_update(
        self, HI: ndarray, s: ndarray, y: ndarray
    ) -> ndarray:

        Id = np.identity(HI.shape[-1])

        rho = 1 / np.dot(y, s)

        G = Id - rho * np.outer(s, y)
        GT = Id - rho * np.outer(y, s)

        H_upd = np.matmul(np.matmul(G, HI), GT) + rho * np.outer(s, s)

        return H_upd

    def step(self):
        HI = np.copy(self.HI)
        r = np.copy(self.r)
        x = np.copy(self.x)

        s = np.copy(x)
        y = np.copy(r)

        p = -1.0 * np.matmul(HI, r)

        x = x + p

        r = self.loss.grad(x)

        s = x - s
        y = r - y

        self.HI = self.hessian_inv_update(HI, s, y)
        self.x = np.copy(x)
        self.r = np.copy(r)

        return None

    def minimize(self):
        k = 0
        gnorm = np.linalg.norm(self.r)

        fstr = open("opt.dat", "w")

        while gnorm > self.gtol and k < self.max_iter:
            self.step()
            gnorm = np.linalg.norm(self.r)
            k = k + 1
            print(k, gnorm, file=fstr)

        fstr.close()

        if self.loss.model is not None:
            self.loss.model.set_params(tc.tensor(self.x))

        return k
