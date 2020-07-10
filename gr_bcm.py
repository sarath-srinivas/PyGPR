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

    def train(self, method='Nelder-Mead', jac=False):
        pass


def sample_with_repulsion(mins, maxs, min_dist, max_count=5000):

    dim = len(mins)
    xc = tc.empty([max_count, dim])

    xc[0, :] = mins + tc.rand(1, dim).mul_(maxs - mins)

    k = tc.tensor(1, dtype=tc.int64)
    count = tc.tensor(1, dtype=tc.int64)

    while k < max_count and count < max_count:

        x = mins + tc.rand(1, dim).mul_(maxs - mins)

        dist = tc.sum((xc[:k, :] - x).square_(), 1).sqrt_()

        if tc.all(dist.sub_(min_dist) > 1E-5):
            xc[k, :] = x
            k.add_(1)
            count.mul_(0)

        count.add_(1)

    return xc[:k, :]


def euclidean_dist(x, y):
    x2 = tc.sum(x.square(), 1)
    y2 = tc.sum(y.square(), 1)

    sqd = -2.0 * tc.matmul(x, y.transpose(0, 1)) + (x2.reshape(-1, 1) +
                                                    y2.reshape(1, -1))

    return sqd.sqrt_()


def cluster_samples(xc, ns, mins, maxs):

    nc = xc.shape[0]
    dim = xc.shape[1]

    xpart = tc.empty([nc, ns, dim])

    x = mins + tc.rand(10 * ns * nc, dim).mul_(maxs - mins)

    dist = euclidean_dist(x, xc)

    idx = tc.argmin(dist, 1)

    for i in range(0, nc):

        xpart[i, :, :] = x[idx == i][:ns, :]

    return xpart
