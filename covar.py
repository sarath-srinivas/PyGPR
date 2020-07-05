import torch as tc
import numpy as np

tc.set_default_tensor_type(tc.DoubleTensor)


def sq_exp(x, xs=None, hp=None, deriv=False, **kargs):

    n = x.shape[0]
    dim = x.shape[1]

    if hp is None:
        hp = tc.ones(dim + 2)
        hp[1] = 1E-4
        return hp
    else:
        hp = tc.tensor(hp)

    ls = hp[2:]
    sig = hp[0]
    sig_noise = hp[1]
    eps = 1e-5 + sig_noise.square()
    xl = x.mul(ls)
    x2 = tc.sum(xl.square(), 1)

    if xs is None:
        sqd = 2.0 * tc.matmul(xl, xl.transpose(0, 1)) - (x2.reshape(-1, 1) +
                                                         x2.reshape(1, -1))
        sqd.exp_()
        sqd.mul_(sig.square())
        sqd.add_(tc.eye(n).mul_(eps))

    else:
        xsl = xs.mul(ls)
        xs2 = tc.sum(xsl.square(), 1)

        sqd = 2.0 * tc.matmul(xsl, xl.transpose(0, 1)) - (xs2.reshape(-1, 1) +
                                                          x2.reshape(1, -1))
        sqd.exp_()
        sqd.mul_(sig.square())

    if deriv == False:
        return sqd
    else:
        return sqd, d_sq_exp(x, sqd, hp)


def d_sq_exp(x, krn, hp):

    nhp = len(hp)
    n = krn.shape[0]

    dkrn = tc.empty([nhp, n, n], dtype=tc.float64)

    krn = krn.sub(tc.eye(n).mul_(hp[1]**2))

    dkrn[0] = krn.mul((2.0 / hp[0]))
    dkrn[1] = tc.eye(n, dtype=tc.float64).mul_(2.0 * hp[1])

    xt = x.transpose(0, 1)
    diff = xt[:, :, np.newaxis] - xt[:, np.newaxis, :]

    diff.square_()
    diff.mul_(hp[2:, np.newaxis, np.newaxis])
    diff.mul_(krn[np.newaxis, :, :])
    diff.mul_(-2.0)

    dkrn[2:] = diff

    return dkrn
