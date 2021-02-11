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


class SK_WRAP(RegressorMixin, BaseEstimator):
    """
     Scikit model wrapper for GPR
    """

    def __init__(self, model: GPR) -> None:
        self.model: GPR = model
        return None

    def fit(self, x: Tensor, y: Tensor) -> Any:
        print("Fitting", x.shape, y.shape)
        self.model.x = x
        self.model.y = y

        return self

    def predict(self, xp: Tensor) -> Tensor:
        print("Predicting", xp.shape)
        self.need_upd = True
        yp, covar = self.model.predict(xp, var="none")
        return yp
