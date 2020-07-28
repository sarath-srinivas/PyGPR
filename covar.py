import torch as tc
import numpy as np

tc.set_default_tensor_type(tc.DoubleTensor)
tc.set_printoptions(precision=7, sci_mode=True)


def dist_square(x, xs=None):

    x = x.view((-1, x.shape[-2], x.shape[-1]))

    x2 = tc.sum(x.square(), 2)

    if xs is None:

        sqd = -2.0 * tc.matmul(x, x.transpose(1, 2)) + x2.unsqueeze(2).add(
            x2.unsqueeze(1))

    else:
        xs = xs.view((-1, xs.shape[-2], xs.shape[-1]))

        xs2 = tc.sum(xs.square(), 2)

        sqd = -2.0 * tc.matmul(xs, x.transpose(1, 2)) + xs2.unsqueeze_(2).add(
            x2.unsqueeze_(1))

        xs.squeeze_(0)

    x.squeeze_(0)

    return sqd.squeeze(0)


def sq_exp_noise(x, xs=None, hp=None, deriv=False, **kargs):

    x = x.view((-1, x.shape[-2], x.shape[-1]))

    dim = x.shape[-1]
    nc = x.shape[0]

    if hp is None:
        hp = tc.ones([nc, dim + 2])
        hp[:, 1].fill_(1E-4)
        hp.squeeze_(0)
        return hp
    elif not tc.is_tensor(hp):
        hp = tc.tensor(hp)
    else:
        pass

    if hp.dim() == 1:
        hp = hp.view((nc, -1))

    assert hp.dim() == 2

    sig = hp[:, 0]
    sig_noise = hp[:, 1]
    ls = hp[:, 2:]
    eps = sig_noise.square()
    ls.unsqueeze_(1)
    xl = x.mul(ls)

    if xs is None:
        sqd = dist_square(xl)

        sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

        sqd.mul_(-1.0)
        sqd.exp_()
        sqd.mul_(sig[:, None, None].square())

        idt = tc.empty_like(sqd).copy_(tc.eye(sqd.shape[-1]))
        idt.mul_(eps[:, None, None])
        sqd.add_(idt)

    else:
        xs = xs.view((-1, xs.shape[-2], xs.shape[-1]))
        xsl = xs.mul(ls)

        sqd = dist_square(xl, xs=xsl)
        sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

        sqd.mul_(-1.0)
        sqd.exp_()
        sqd.mul_(sig[:, None, None].square())

    hp.squeeze_(0)

    sqd.squeeze_(0)

    if deriv == False:
        return sqd
    else:
        return sqd, d_sq_exp_noise(x, sqd, hp)


def d_sq_exp_noise(x, krn, hp):

    x = x.view((-1, x.shape[-2], x.shape[-1]))
    krn = krn.view((-1, krn.shape[-2], krn.shape[-1]))

    nc = x.shape[0]
    nhp = hp.shape[-1]
    n = krn.shape[-1]

    if hp.dim() == 1:
        hp = hp.view((nc, -1))

    assert hp.dim() == 2

    dkrn = tc.empty([nc, nhp, n, n], dtype=tc.float64)

    sig = hp[:, 0]
    sig_noise = hp[:, 1]
    ls = hp[:, 2:]
    eps = sig_noise.square()

    idt = tc.empty_like(krn).copy_(tc.eye(n))
    krn_noise = idt.mul(eps[:, None, None])
    krn = krn.sub(krn_noise)

    dkrn[:, 0, :, :] = krn.mul(sig[:, None, None].reciprocal().mul_(2.0))
    dkrn[:, 1, :, :] = idt.mul_(sig_noise[:, None, None].mul(2.0))

    xt = x.transpose(-2, -1)
    diff = xt[:, :, :, None].sub(xt[:, :, None, :])

    diff.square_()
    diff.mul_(ls[:, :, None, None])
    diff.mul_(krn[:, None, :, :])
    diff.mul_(-2.0)

    dkrn[:, 2:, :, :] = diff

    x.squeeze_(0)
    hp.squeeze_(0)
    dkrn.squeeze_(0)

    return dkrn


def sq_exp(x, xs=None, hp=None, deriv=False, **kargs):

    x = x.view((-1, x.shape[-2], x.shape[-1]))

    dim = x.shape[-1]
    nc = x.shape[0]

    if hp is None:
        hp = tc.ones([nc, dim + 1])
        hp.squeeze_(0)
        return hp
    elif not tc.is_tensor(hp):
        hp = tc.tensor(hp)
    else:
        pass

    if hp.dim() == 1:
        hp = hp.view((nc, -1))

    assert hp.dim() == 2

    sig = hp[:, 0]
    ls = hp[:, 1:]
    eps = 1e-5 * tc.ones(nc)
    ls.unsqueeze_(1)
    xl = x.mul(ls)

    if xs is None:
        sqd = dist_square(xl)

        sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

        sqd.mul_(-1.0)
        sqd.exp_()
        sqd.mul_(sig[:, None, None].square())

        idt = tc.empty_like(sqd).copy_(tc.eye(sqd.shape[-1]))
        idt.mul_(eps[:, None, None])
        sqd.add_(idt)

    else:
        xs = xs.view((-1, xs.shape[-2], xs.shape[-1]))
        xsl = xs.mul(ls)

        sqd = dist_square(xl, xs=xsl)
        sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

        sqd.mul_(-1.0)
        sqd.exp_()
        sqd.mul_(sig[:, None, None].square())

    hp.squeeze_(0)

    sqd.squeeze_(0)

    if deriv == False:
        return sqd
    else:
        return sqd, d_sq_exp(x, sqd, hp)


def d_sq_exp(x, krn, hp):

    x = x.view((-1, x.shape[-2], x.shape[-1]))
    krn = krn.view((-1, krn.shape[-2], krn.shape[-1]))

    nc = x.shape[0]
    nhp = hp.shape[-1]
    n = krn.shape[-1]

    if hp.dim() == 1:
        hp = hp.view((nc, -1))

    assert hp.dim() == 2

    dkrn = tc.empty([nc, nhp, n, n], dtype=tc.float64)

    sig = hp[:, 0]
    ls = hp[:, 1:]
    eps = 1e-5 * tc.ones(nc)

    idt = tc.empty_like(krn).copy_(tc.eye(n))
    idt.mul_(eps[:, None, None])
    krn = krn.sub(idt)

    dkrn[:, 0, :, :] = krn.mul(sig[:, None, None].reciprocal().mul_(2.0))

    xt = x.transpose(-2, -1)

    diff = xt[:, :, :, None].sub(xt[:, :, None, :])

    diff.square_()
    diff.mul_(ls[:, :, None, None])
    diff.mul_(krn[:, None, :, :])
    diff.mul_(-2.0)

    dkrn[:, 1:, :, :] = diff

    x.squeeze_(0)
    hp.squeeze_(0)
    dkrn.squeeze_(0)

    return dkrn


covars = {'sq_exp': sq_exp, 'sq_exp_noise': sq_exp_noise}
