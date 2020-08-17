import torch as tc
import numpy as np
tc.set_default_tensor_type(tc.DoubleTensor)


class UNIFORM(object):

    def __init__(self, seed):
        self.seed = seed

    def sample(self, n, mins, maxs):
        tc.manual_seed(self.seed)
        dim = len(mins)
        x = mins + tc.rand(n, dim).mul_(maxs - mins)
        return x


class MATERN1(UNIFORM):

    def __init__(self, seed):
        super().__init__(seed)
        self.min_dist = None
        self.max_count = 5000

    def sample_repulsion(self, mins, maxs, min_dist):

        tc.manual_seed(self.seed)

        dim = len(mins)
        xc = tc.empty([self.max_count, dim])

        xc[0, :] = mins + tc.rand(1, dim).mul_(maxs - mins)

        k = 1
        count = 1

        while k < self.max_count and count < self.max_count:

            x = mins + tc.rand(1, dim).mul_(maxs - mins)

            dist = tc.sum((xc[:k, :] - x).square_(), 1).sqrt_()

            if tc.all(dist.sub_(min_dist) > 1E-5):
                xc[k, :] = x
                k += 1
                count *= 0

            count += 1

        return xc[:k, :]

    def sample(self, n, mins, maxs):
        vol = tc.prod(maxs - mins)
        dim = len(mins)
        min_dist = (vol / n)**(1 / dim)

        xc = self.sample_repulsion(mins, maxs, min_dist)

        while xc.shape[0] < n:
            min_dist *= 0.9
            xc = self.sample_repulsion(mins, maxs, min_dist)

        self.min_dist = min_dist

        return xc[:n, :]

    def cluster_samples(self, xc, ns, mins, maxs):

        tc.manual_seed(self.seed)

        nc = xc.shape[0]
        dim = xc.shape[1]

        xpart = tc.empty([nc, ns, dim])

        x = mins + tc.rand(10 * ns * nc, dim).mul_(maxs - mins)

        dist = euclidean_dist(x, xc)

        idx = tc.argmin(dist, 1)

        for i in range(0, nc):

            xpart[i, :, :] = x[idx == i][:ns, :]

        return xpart

    def partition(self, nc, ns, mins, maxs):

        xc = self.sample(nc, mins, maxs)
        x = self.cluster_samples(xc, ns, mins, maxs)

        return x, xc


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


def sample_gp(x, cov, hp=None, mean=None, **kwargs):

    if hp is None:
        hp = cov(x)

    krn = cov(x, hp=hp, **kwargs)
    krn_chd = tc.cholesky(krn, upper=False)

    N = tc.randn(x.shape[-2])

    if mean is None:
        f = tc.mv(krn_chd, N)
    else:
        f = tc.mv(krn_chd, N).add_(mean)

    return f
