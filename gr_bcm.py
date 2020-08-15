import torch as tc
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import opt_einsum as oen
from .gpr import GPR, log_likelihood, jac_log_likelihood

tc.set_default_tensor_type(tc.DoubleTensor)


class GRBCM(GPR):
    def __init__(self, xl, yl, xg, yg, cov, hp=None, **kargs):

        self.xg = tc.clone(xg)
        self.yg = tc.clone(yg)
        self.cov = cov
        self.args = kargs
        self.need_upd = True
        self.need_upd_g = True

        nc = xl.shape[0]
        nls = xl.shape[1]
        ng = self.xg.shape[0]
        dim = self.xg.shape[1]

        self.nc = nc
        self.nsc = nls
        self.ng = ng
        self.dim = dim

        if hp is None:
            self.hpg = self.cov(xg)
            self.hp = tc.empty([nc, len(self.hpg)])
            self.hp.copy_(self.hpg)
        else:
            self.hp.copy_(hp)

        tmpx = tc.empty([nc, ng, dim])
        tmpy = tc.empty([nc, ng])

        tmpx.copy_(self.xg)
        tmpy.copy_(self.yg)

        self.x = tc.cat((tmpx, xl), dim=1)
        self.y = tc.cat((tmpy, yl), dim=1)

    def cost_fun_global(self, hp):
        llhd = log_likelihood(self.xg, self.yg, hp, self.cov, **self.args)
        return llhd

    def jac_cost_fun_global(self, hp):
        jac_llhd = jac_log_likelihood(self.xg, self.yg, hp, self.cov,
                                      **self.args)
        return jac_llhd

    def cost_fun_local(self, hp, cen):
        llhd = log_likelihood(self.x[cen, :, :], self.y[cen, :], hp, self.cov,
                              **self.args)
        return llhd

    def jac_cost_fun_local(self, hp, cen):
        jac_llhd = jac_log_likelihood(self.x[cen, :, :], self.y[cen, :], hp,
                                      self.cov, **self.args)
        return jac_llhd

    def train(self, method='CG', jac=True):

        self.llhd = tc.empty(self.x.shape[0])
        self.jac_llhd = tc.empty_like(self.hp)

        if jac:
            for cen in range(0, self.nc):
                res = opt.minimize(self.cost_fun_local,
                                   self.hp[cen, :],
                                   args=(cen, ),
                                   jac=self.jac_cost_fun_local,
                                   method=method)
                self.hp[cen, :] = tc.tensor(res.x)
                self.llhd[cen] = tc.tensor(res.fun)
                self.jac_llhd[cen, :] = tc.tensor(res.jac)

            self.need_upd = True

            res = opt.minimize(self.cost_fun_global,
                               self.hpg,
                               jac=self.jac_cost_fun_global,
                               method=method)
            self.hpg = tc.tensor(res.x)
            self.llhd_g = tc.tensor(res.fun)
            self.jac_llhd_g = tc.tensor(res.jac)
            self.need_upd_g = True

            return res

        else:
            for cen in range(0, self.nc):
                res = opt.minimize(self.cost_fun_local,
                                   self.hp[cen, :],
                                   args=(cen, ),
                                   jac=False,
                                   method=method)
                self.hp[cen, :] = tc.tensor(res.x)
                self.llhd[cen] = tc.tensor(res.fun)
                self.jac_llhd[cen, :] = tc.tensor(res.jac)

            self.need_upd = True

            res = opt.minimize(self.cost_fun_global,
                               self.hpg,
                               jac=False,
                               method=method)
            self.hpg = tc.tensor(res.x)
            self.llhd_g = tc.tensor(res.fun)
            self.jac_llhd_g = tc.tensor(res.jac)
            self.need_upd_g = True

            return res

    def interpolate_global(self, xs):
        if self.need_upd_g:
            self.krng = self.cov(self.xg, hp=self.hpg, **self.args)
            self.krnchdg = tc.cholesky(self.krng)
            self.wtg = tc.squeeze(
                tc.cholesky_solve(self.yg.reshape(-1, 1), self.krnchdg))
            self.need_upd_g = False

        krns = self.cov(self.xg, xs=xs, hp=self.hpg, **self.args)
        ys = tc.mv(krns, self.wtg)

        covars = super().get_pred_covar(xs, krns, krnchd=self.krnchdg, hp=self.hpg)

        return ys, covars

    def interpolate_local(self, xs):

        if self.need_upd:
            self.krn = self.cov(self.x, hp=self.hp, **self.args)
            self.krnchd = tc.cholesky(self.krn)
            y = self.y.view(-1, self.y.shape[-1], 1)
            self.wt = tc.cholesky_solve(y, self.krnchd)
            self.need_upd = False

        krns = self.cov(self.x, xs=xs, hp=self.hp, **self.args)
        ys = tc.bmm(krns, self.wt)
        ys.squeeze_(-1)
        krnss = self.cov(xs, hp=self.hp, **self.args)
        lks = tc.cholesky_solve(krns.transpose(1, 2), self.krnchd)
        covars = krnss.sub_(tc.bmm(krns, lks))

        return ys, covars

    def aggregate_covar(self, beta, covars_g, covars_l):
        covar_gl = tc.cat((covars_g[None, :, :], covars_l))
        covar_gl_chd = tc.cholesky(covar_gl)
        idt = tc.empty_like(covar_gl).copy_(tc.eye(covar_gl.shape[-1]))
        prec_gl = tc.cholesky_solve(idt, covar_gl_chd)

        prec = oen.contract('c,cij->ij', beta, prec_gl, backend='torch')
        covars = tc.cholesky_inverse(tc.cholesky(prec))

        return covars

    def aggregate(self, ys_g, covars_g, ys_l, covars_l, diag_only=True):

        beta = tc.empty(self.nc + 1, ys_g.shape[-1])
        prec = tc.empty(self.nc + 1, ys_g.shape[-1])

        print(prec.shape)

        prec[0, :] = tc.diag(covars_g).reciprocal_()
        prec[1:, :] = tc.diagonal(covars_l, dim1=-2, dim2=-1).reciprocal()

        beta[1:, :] = tc.log(prec[1:, :]).sub_(tc.log(prec[0, :])).mul_(0.5)
        beta[1, :].fill_(1.0)
        beta[0, :] = beta[1:, :].sum(0).sub_(1.0).mul_(-1.0)

        self.beta = tc.clone(beta)
        self.prec = tc.clone(prec)

        ys = tc.cat((ys_g[None, :], ys_l))

        precs = prec.mul_(beta)

        if diag_only:
            covars = precs.sum(0).reciprocal_()
            ys = ys.mul_(precs).sum(0).mul_(covars)
            covars = tc.diag(covars)
        else:
            covars = self.aggregate_covar(beta, covars_g, covars_l)
            ys = ys.mul_(precs).sum(0).mul_(tc.diag(covars))

        return ys, covars

    def interpolate(self, xs, diag_only=True):
        ys_g, covars_g = self.interpolate_global(xs)
        ys_l, covars_l = self.interpolate_local(xs)

        return self.aggregate(ys_g, covars_g, ys_l, covars_l, diag_only=diag_only)

    def plot_hparam(self, ax=None):
        cno = range(0, self.nc)
        for i in range(0, self.dim):
            ax.scatter(cno, self.hp[:, i], label='length scale {}'.format(i))
        ax.set(xlabel='Centre no.')
        ax.legend()

    def plot_jac(self, ax=None):
        cno = range(0, self.nc)
        for i in range(0, self.dim):
            ax.scatter(cno, self.jac_llhd[:, i], label='dL/dls {}'.format(i))
        ax.set(xlabel='Centre no.')
        ax.legend()


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
