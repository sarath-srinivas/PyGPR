import torch as tc
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from .gpr import GPR

tc.set_default_tensor_type(tc.DoubleTensor)


class GRBCM(GPR):
    def __init__(self, xl, yl, xg, yg, cov, hp=None, **kargs):

        super().__init__(xl, yl, cov, hp, **kargs)

        self.xg = tc.clone(xg)
        self.yg = tc.clone(yg)

        nc = xl.shape[0]
        nls = xl.shape[1]
        ng = self.xg.shape[0]
        dim = self.xg.shape[1]

        if hp is None:
            tmp = self.hp
            self.hp = tc.empty([nc, len(tmp)])
            self.hp.copy_(tmp)
        else:
            self.hp.copy_(hp)

        tmpx = tc.empty([nc, ng, dim])
        tmpy = tc.empty([nc, ng])

        tmpx.copy_(self.xg)
        tmpy.copy_(self.yg)

        self.x = tc.cat((tmpx, xl), dim=1)
        self.y = tc.cat((tmpy, yl), dim=1)

    def cost_fun_global(self, hp):
        llhd = log_likelihood(self.x, self.y, hp, self.cov, **selg.args)
        return llhd.sum()

    def cost_fun_local(self, hp, cen):
        llhd = log_likelihood(self.x[cen, :, :], self.y[cen, :], hp, self.cov,
                **selg.args)
        return llhd

    def train_local(self, method='Nelder-Mead', jac=False):

        self.llhd = tc.empty(self.x.shape[0])
        self.jac_llhd = tc.empty_like(self.hp)

        if jac:
            for cen in range(0,self.x.shape[0]):
                res = opt.minimize(self.cost_fun_local,
                        self.hp[cen, :], args=(cen,),
                        jac=jac_cost_fun_local,
                        method=method)

                self.hp[cen,:] = res.x
                self.llhd[cen] = res.fun
                self.jac_llhd[cen,:] = res.jac

        else:
            for cen in range(0,self.x.shape[0]):
                res = opt.minimize(self.cost_fun_local,
                        self.hp[cen, :], args=(cen,),
                        jac=False,
                        method=method)

                self.hp[cen,:] = res.x
                self.llhd[cen] = res.fun
                self.jac_llhd[cen,:] = res.jac

        return res

    def train_global(self, method='Nelder-Mead', jac=False)

        if jac:
            res = opt.minimize(self.cost_fun_global,
                    self.hp,
                    jac=jac_cost_fun_global,
                    method=method)
        else:
            res = opt.minimize(self.cost_fun_global,
                    self.hp,
                    jac=False,
                    method=method)

            self.hp = res.x
        self.llhd= res.fun
        self.jac_llhd = res.jac

        return res


def log_likelihood(x, y, hp, cov, **kwargs):

    krn = cov(x, hp=hp, **kwargs)
    krnchd = tc.cholesky(krn)

    y = y.view(-1, y.shape[-1], 1)

    wt = tc.cholesky_solve(y, krnchd)

    wt.squeeze_(2)
    y.squeeze_(2)

    llhd = 0.5 * wt.mul_(y).sum(-1) \
            + tc.log(tc.diagonal(krnchd, dim1=-2, dim2=-1)).sum(-1) \
            + 0.5 * y.shape[-1] * tc.log(tc.tensor(2 * np.pi))

    llhd.squeeze_(0)

    return llhd.numpy()
