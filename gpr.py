import torch as tc
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

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

        self.dgn = {}

    def train(self, method='Nelder-Mead'):

        res = opt.minimize(log_likelihood,
                           self.hp,
                           args=(self.cov, self.x, self.y),
                           method=method)
        self.hp = res.x
        self.llhd = res.fun
        self.jac_llhd = jac_log_likelihood(self.hp, self.cov, self.x, self.y,
                                           **self.args)

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

    def diagnostics(self, xs, ys, covar, ya, diag=False):
        var = tc.diag(covar)
        n = ys.shape[0]
        err = ys - ya

        self.dgn['RMSE'] = tc.sqrt(tc.mean(tc.sum(err**2)))
        self.dgn['SDSUM'] = tc.sqrt(tc.mean(tc.sum(var)))
        self.dgn['RCHI-SQ'] = (1.0 / n) * tc.sum((err**2) / var)

        if diag == True:
            self.dgn['LLHD'] = -0.5 * tc.sum(np.log(var)) \
                    - 0.5 * tc.log( 2 * np.pi) - n * self.dgn['RCHI-SQ']
        else:
            eig, evec = tc.symeig(covar)
            sol, lu = tc.solve(err[:, np.newaxis], covar)
            md = tc.dot(err, sol.squeeze_())
            self.dgn['LLHD'] = -0.5 * tc.sum(tc.log(eig)) \
                    - 0.5 * tc.log(tc.tensor(2 * np.pi)) - md
            self.dgn['MD'] = (1.0 / n) * md

    def plot(self, xs, ys, covars, ya, diag=False):
        if diag:
            sig = tc.sqrt(covars)
        else:
            sig = tc.sqrt(tc.diag(covars))

        min_ys = tc.min(ys)
        max_ys = tc.max(ys)

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].scatter(ys, ya, color='red')
        ax[0, 0].plot([min_ys, max_ys], [min_ys, max_ys])
        ax[0, 0].axis('equal')
        ax[0, 0].set(title='Prediction Vs Exact',
                     xlabel='Y Predicted',
                     ylabel='Y actual')

        ax[0, 1].hist(tc.log(sig))
        ax[0, 1].set(title='$\sigma$-Predicted',
                     xlabel='$log(\sigma)$',
                     ylabel='Frequency')

        ax[1, 0].scatter(range(0, len(self.hp)), self.hp, label='$\\theta$')
        ax[1, 0].scatter(range(0, len(self.hp)),
                         self.jac_llhd,
                         label='$dL/d\\theta$')
        ax[1, 0].set(title='Derivative and hyperparam', xlabel='S.No')
        ax[1, 0].legend()

        ax[1, 1].hist(tc.log(ys - ya))
        ax[1, 1].set(title='Mean Squared Error',
                     xlabel='MSE',
                     ylabel='Frequency')


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
