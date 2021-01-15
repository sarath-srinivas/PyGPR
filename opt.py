import torch as tc
import numpy as np
from numpy import ndarray
from .loss import Loss
import scipy.optimize as scopt

tc.set_default_tensor_type(tc.DoubleTensor)


class Opt:
    """
     Base class for optimisers.
    """

    def __init__(self, loss: Loss) -> None:
        self.loss: Loss = loss
        self.args: dict = {}
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
