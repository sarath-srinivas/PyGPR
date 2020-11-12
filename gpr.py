import torch as tc
from torch import Tensor
from typing import Sequence
from covar import Covar, Squared_exponential
from loss import Loss, MLE
from opt import Opt, CG
from var import Var, Exact_var

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR():
    """
     Class for Gaussian process regression.
    """

    def __init__(self, x: Tensor, y: Tensor, cov: Covar = None,
                 loss: Loss = None, opt: Opt = None, var: Var = None) -> None:

        self.x: Tensor = x
        self.y: Tensor = y

        if cov is None:
            cov = Squared_exponential(x)
        self.cov: Covar = cov

        if loss is None:
            loss = MLE(x, y, self.cov)

        if opt is None:
            opt = CG()

        if var is None:
            var = Exact_var()

        self.loss: Loss = loss
        self.opt: Opt = opt
        self.var: Var = var
        self.params: Tensor = self.params

        self.krn: Tensor = NotImplemented
        self.wt: Tensor = NotImplemented
        self.krnchd: Tensor = NotImplemented

        self.need_upd: bool = True

        return None

    def train(self) -> None:

        self.opt.minimize(self.loss.loss_grad(self.params))
        return None

    def update(self) -> None:
        if self.need_upd:
            self.krn = self.cov.kernel(self.x)
            self.krnchd = tc.cholesky(self.krn)
            self.wt = tc.cholesky_solve(self.y[..., None],
                                        self.krnchd).squeeze_(-1)
            self.need_upd = False
        return None

    def predict(self, xp: Tensor, diag_only: bool = False) -> Sequence[Tensor]:

        self.update()
        krns = self.cov.kernel(self.x, xp=xp)
        ys = tc.bmm(krns.view(-1, *krns.shape[-2:]),
                    self.wt.view(-1, self.wt.shape[-1], 1))

        ys = ys.squeeze_()

        if diag_only is True:
            covars = self.var.pred_covar(xp, self, krns=krns)
        else:
            covars = self.var.pred_var(xp, self, krns=krns)

        return [ys, covars]
