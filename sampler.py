import torch as tc
import numpy as np


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


def nsample_repulsion(nc, mins, maxs, max_count=5000):
    vol = tc.prod(maxs - mins)
    dim = len(mins)
    min_dist = (vol / nc)**(1 / dim)

    xc = sample_with_repulsion(mins, maxs, min_dist, max_count=max_count)

    while xc.shape[0] < nc:
        min_dist *= 0.9
        xc = sample_with_repulsion(mins, maxs, min_dist, max_count=max_count)

    return xc[:nc, :], min_dist


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
