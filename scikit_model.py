from torch import Tensor
from .gpr import GPR
from .loss import Loss
from .opt import Opt
from .covar import Covar
from typing import Callable, Any
from sklearn.base import BaseEstimator, RegressorMixin

T_GPR = Callable[..., GPR]
T_LOSS = Callable[..., Loss]
T_OPT = Callable[..., Opt]
T_COVAR = Callable[..., Covar]


class SK_GPR(RegressorMixin, BaseEstimator):
    """
     Scikit model wrapper for GPR
    """

    def __init__(
        self, model: T_GPR, covar: T_COVAR, loss: T_LOSS, opt: T_OPT
    ) -> None:
        self.model: T_GPR = model
        self.loss: T_LOSS = loss
        self.opt: T_OPT = opt
        self.covar: Covar = covar
        # self.mod: GPR = NotImplemented
        return None

    def fit(self, x: Tensor, y: Tensor) -> Any:
        print("Fitting")
        self.mod = self.model(x, y, self.covar())
        loss = self.loss(self.mod)
        opt = self.opt(loss)

        opt.minimize()

        return self

    def predict(self, xp: Tensor) -> Tensor:
        print("Predicting")
        yp, covar = self.mod.predict(xp, var="none")
        return yp
