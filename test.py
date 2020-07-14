import torch as tc
import numpy as np
import scipy as scp
from .covar import covars

tc.set_default_tensor_type(tc.DoubleTensor)


class TEST(object):
    def __init__(self, n=100, dim=2):
        self.covar = {}
        self.covar_err = {}
        self.n = n
        self.dim = dim

    def covar_test(self, eps=1e-5, tol=1e-5):

        x = tc.rand([self.n, self.dim])
        self.covar['symmetric'] = {}
        self.covar['pos_def'] = {}
        self.covar['derivative'] = {}
        self.covar_err['symmetric'] = {}
        self.covar_err['pos_def'] = {}
        self.covar_err['derivative'] = {}

        for key in covars:

            cov = covars[key]
            hp = tc.rand(cov(x).shape[0])
            krn = cov(x, hp=hp)

            self.covar['symmetric'][key] = tc.allclose(krn, krn.t(), atol=tol)
            self.covar_err['symmetric'][key] = tc.sub(krn, krn.t())

            eig = tc.eig(krn)[0][:, 0]

            self.covar['pos_def'][key] = tc.all(eig > 0)

            self.covar_err['pos_def'][key] = eig

            eps = tc.tensor(1E-5)

            #dkrn_num = tc.empty([len(hp), self.n, self.n])

            #for p in range(0, len(hp)):

            #    hp_eps = hp.index_add(0, tc.tensor(p), eps)
            #    dkrn_num[p, :, :] = cov(x, hp=hp_eps).sub_(cov(x, hp=hp))
            #    dkrn_num[p, :, :].div_(eps)

            #dkrn = cov(x, hp=hp, deriv=True)[1]

            #self.covar['derivative'][key] = tc.allclose(dkrn_num,
            #                                            dkrn,
            #                                            atol=tol)

            #self.covar_err['derivative'][key] = dkrn_num.sub_(dkrn)
