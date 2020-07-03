import torch as tc
import numpy as np


def sq_exp(x, xs=None, hp=None):

    n = x.shape[0]
    dim = x.shape[1]

    if hp is None:
        hp = tc.ones(dim + 2)
        hp[1] = 1e-5

    ls = hp[2:]
    sig = hp[0]
    eps = hp[1]
    xl = x.mul(ls)
    x2 = tc.sum(xl.square(), 1)

    if xs is None:
        sqd = 2.0 * tc.matmul(xl, xl.transpose(0, 1)) - (x2.reshape(-1, 1) +
                                                         x2.reshape(1, -1))
        sqd.exp_()
        sqd.mul_(sig)
        sqd.add_(eps * tc.eye(n))

    else:
        xsl = xs.mul(ls)
        xs2 = tc.sum(xsl.square(), 1)

        sqd = 2.0 * tc.matmul(xsl, xl.transpose(0, 1)) - (xs2.reshape(-1, 1) +
                                                          x2.reshape(1, -1))
        sqd.exp_()
        sqd.mul_(sig)

    return sqd
