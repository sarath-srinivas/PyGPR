import torch as tc
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import opt_einsum as oen
from .gpr import GPR, Exact_GP

tc.set_default_tensor_type(tc.DoubleTensor)


class GRBCM(GPR):
    def __init__(self, xl, yl, xg, yg, cov, hp=None, **kargs):

        nc = xl.shape[0]
        nls = xl.shape[1]
        ng = xg.shape[0]
        dim = xg.shape[1]

        tmpx = tc.empty([nc, ng, dim])
        tmpy = tc.empty([nc, ng])

        tmpx.copy_(xg)
        tmpy.copy_(yg)

        x = tc.cat((tmpx, xl), dim=1)
        y = tc.cat((tmpy, yl), dim=1)

        self.gpg = Exact_GP(xg, yg, cov)
        self.gpl = Exact_GP(x, y, cov)

        self.nc = nc
        self.nsc = nls
        self.ng = ng
        self.dim = dim

    def cost_fun_global(self, hp):
        return self.gpg.cost_fun(hp)

    def jac_cost_fun_global(self, hp):
        return self.gpg.jac_cost_fun(hp)

    def cost_fun_local(self, hp, cen):
        llhd = log_likelihood(self.gpl.x[cen, :, :], self.gpl.y[cen, :], hp, self.gpl.cov,
                              **self.gpl.args)
        return llhd

    def jac_cost_fun_local(self, hp, cen):
        jac_llhd = jac_log_likelihood(self.gpl.x[cen, :, :], self.gpl.y[cen, :], hp,
                                      self.gpl.cov, **self.gpl.args)
        return jac_llhd

    def train(self, method='CG', jac=True):

        self.gpl.llhd = tc.empty(self.gpl.x.shape[0])
        self.gpl.jac_llhd = tc.empty_like(self.gpl.hp)

        if jac:
            for cen in range(0, self.nc):
                res = opt.minimize(self.cost_fun_local,
                                   self.gpl.hp[cen, :],
                                   args=(cen, ),
                                   jac=self.jac_cost_fun_local,
                                   method=method)
                self.gpl.hp[cen, :] = tc.tensor(res.x)
                self.gpl.llhd[cen] = tc.tensor(res.fun)
                self.gpl.jac_llhd[cen, :] = tc.tensor(res.jac)

            res = opt.minimize(self.cost_fun_global,
                               self.gpg.hp,
                               jac=self.jac_cost_fun_global,
                               method=method)
            self.gpg.hp = tc.tensor(res.x)
            self.gpg.llhd = tc.tensor(res.fun)
            self.gpg.jac_llhd = tc.tensor(res.jac)

            return res

        else:
            for cen in range(0, self.nc):
                res = opt.minimize(self.cost_fun_local,
                                   self.gpl.hp[cen, :],
                                   args=(cen, ),
                                   jac=False,
                                   method=method)
                self.gpl.hp[cen, :] = tc.tensor(res.x)
                self.gpl.llhd[cen] = tc.tensor(res.fun)
                self.gpl.jac_llhd[cen, :] = tc.tensor(res.jac)

            res = opt.minimize(self.cost_fun_global,
                               self.gpg.hp,
                               jac=False,
                               method=method)
            self.gpg.hp = tc.tensor(res.x)
            self.gpg.llhd = tc.tensor(res.fun)
            self.gpg.jac_llhd = tc.tensor(res.jac)

            return res

    def aggregate_full_covar(self, beta, covars_g, covars_l):
        covar_gl = tc.cat((covars_g[None, :, :], covars_l))
        covar_gl_chd = tc.cholesky(covar_gl)
        idt = tc.empty_like(covar_gl).copy_(tc.eye(covar_gl.shape[-1]))
        prec_gl = tc.cholesky_solve(idt, covar_gl_chd)

        beta_covar = tc.empty_like(prec_gl)

        beta_covar = beta[:, :, None].add(beta[:, None, :])
        beta_covar.mul_(0.5)

        #prec = oen.contract('c,cij->ij', beta, prec_gl, backend='torch')
        prec = prec_gl.mul_(beta_covar).sum(0)
        covars = tc.cholesky_inverse(tc.cholesky(prec))

        return covars

    def aggregate(self, ys_g, covars_g, ys_l, covars_l, var="diag"):

        if var == "diag":
            var_g = covars_g
            var_l = covars_l
        else:
            var_g = tc.diag(covars_g)
            var_l = tc.diagonal(covars_l, dim1=-2, dim2=-1)

        beta = tc.empty(self.nc + 1, ys_g.shape[-1])
        prec = tc.empty(self.nc + 1, ys_g.shape[-1])

        prec[0, :] = var_g.reciprocal_()
        prec[1:, :] = var_l.reciprocal()

        beta[1:, :] = tc.log(prec[1:, :]).sub_(tc.log(prec[0, :])).mul_(0.5)
        beta[1, :].fill_(1.0)
        beta[0, :] = beta[1:, :].sum(0).sub_(1.0).mul_(-1.0)

        self.beta = tc.clone(beta)
        self.prec = tc.clone(prec)

        ys = tc.cat((ys_g[None, :], ys_l))

        precs = prec.mul_(beta)

        if var=='diag':
            covars = precs.sum(0).reciprocal_()
            ys = ys.mul_(precs).sum(0).mul_(covars)
        else:
            covars = self.aggregate_full_covar(beta, covars_g, covars_l)
            ys = ys.mul_(precs).sum(0).mul_(tc.diag(covars))

        return ys, covars

    def predict(self, xs, var="diag"):
        ys_g, covars_g = self.gpg.predict(xs, var=var)
        ys_l, covars_l = self.gpl.predict(xs, var=var)

        return self.aggregate(ys_g, covars_g, ys_l, covars_l, var=var)


def log_likelihood_batched(x, y, hp, cov, **kwargs):

    krn = cov(x, hp=hp, **kwargs)
    krnchd = tc.cholesky(krn)

    y = y.view(-1, y.shape[-1], 1)

    wt = tc.cholesky_solve(y, krnchd)

    wt.squeeze_(2)
    y.squeeze_(2)

    llhd = 0.5 * wt.mul_(y).sum(-1)
    + tc.log(tc.diagonal(krnchd, dim1=-2, dim2=-1)).sum(-1)
    + 0.5 * y.shape[-1] * tc.log(tc.tensor(2 * np.pi))

    llhd.squeeze_(0)

    return llhd.numpy()
