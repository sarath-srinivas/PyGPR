import torch as tc
import numpy as np
import scipy.optimize as opt

tc.set_default_tensor_type(tc.DoubleTensor)


class GPR(object):
    def __init__(self, x, y, cov, hp=None, **kargs):

        self.x = tc.clone(x)
        self.y = tc.clone(y)
        self.cov = cov
        if hp is None:
            self.hp = self.cov(x)
        else:
            self.hp = hp
        self.args = kargs

        self.krn = NotImplemented
        self.wt = NotImplemented
        self.krnchd = NotImplemented

    def train(self, method='Nelder-Mead'):

        res = opt.minimize(log_likelihood,
                           self.hp,
                           args=(self.cov, self.x, self.y),
                           method=method)
        self.hp = res.x

        return res

    def interpolant(self):

        self.krn = self.cov(self.x, hp=self.hp, **self.args)
        self.krnchd = tc.cholesky(self.krn)
        self.wt = tc.squeeze(
            tc.cholesky_solve(self.y.reshape(-1, 1), self.krnchd))

        def interp_fun(xs):
            krns = self.cov(self.x, xs=xs, hp=self.hp, **self.args)

            ys = tc.mv(krns, self.wt)

            krnss = self.cov(xs, hp=self.hp, **self.args)

            lks = tc.cholesky_solve(krns.transpose(0, 1), self.krnchd)

            covars = krnss - tc.mm(krns, lks)

            return ys, covars

        return interp_fun


def log_likelihood(hp, *args, **kwargs):
    cov = args[0]

    x = args[1]
    y = args[2]

    krn = cov(x, hp=hp, **kwargs)
    krnchd = tc.cholesky(krn)

    wt = tc.squeeze(tc.cholesky_solve(y.reshape(-1, 1), krnchd))

    llhd = 0.5 * tc.dot(wt, y) \
            + tc.sum(tc.log(tc.diag(krnchd))) \
            + 0.5 * len(y) * tc.log(tc.tensor(2 * np.pi))

    return llhd.numpy()


def jac_log_likelihood(hp, *args, **kwargs):
    cov = args[0]
    x = args[1]
    y = args[2]

    krn, dkrn = cov(x, hp=hp, deriv=True, **kwargs)
    krnchd = tc.cholesky(krn)

    wt = tc.cholesky_solve(y.reshape(-1, 1), krnchd).squeeze_()

    jac_llhd = dkrn.matmul(wt).matmul(wt)
    jac_llhd.sub_(tc.sum(tc.diagonal(dkrn, dim1=-1, dim2=-2), 1))
    jac_llhd.mul_(-0.5)

    return jac_llhd.numpy()
