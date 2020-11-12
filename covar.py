import torch as tc
from torch import Tensor
from typing import List, Dict

tc.set_default_tensor_type(tc.DoubleTensor)
tc.set_printoptions(precision=7, sci_mode=True)


class Covar():
    """
     Base class for covariance kernels for Gaussian process
     regression.
    """

    def __init__(self) -> None:
        self.params: Tensor = NotImplemented
        self.params_dict: Dict[str, Tensor] = {}

    def kernel(self, x: Tensor, xp: Tensor = None) -> Tensor:
        raise NotImplementedError

    def kernel_and_grad(self, x: Tensor) -> List[Tensor]:
        raise NotImplementedError


class Squared_exponential(Covar):
    '''
     Squared exponential covariance K(x,x') = sig_y * exp(-|(x-x').ls|^2)
    '''

    def __init__(self, x: Tensor) -> None:
        super().__init__()
        xb = x.view((-1, x.shape[-2], x.shape[-1]))
        dim = xb.shape[-1]
        nc = xb.shape[0]
        self.params: Tensor = tc.ones([nc, dim+2])
        self.params_dict["sig_y"] = self.params[:, 0].squeeze_(0)
        self.params_dict["sig_noise"] = self.params[:, 1].squeeze_(0)
        self.params_dict["ls"] = self.params[:, 2:].squeeze_(0)
        self.params.squeeze_(0)

    def distance(self, x: Tensor, xp: Tensor = None) -> Tensor:
        x = x.view((-1, x.shape[-2], x.shape[-1]))

        x2 = tc.sum(x.square(), 2)

        if xp is None:

            sqd = -2.0 * tc.matmul(x, x.transpose(1, 2)) \
                 + x2.unsqueeze(2).add(x2.unsqueeze(1))

        else:
            xp = xp.view((-1, xp.shape[-2], xp.shape[-1]))

            xp2 = tc.sum(xp.square(), 2)

            sqd = -2.0 * tc.matmul(xp, x.transpose(1, 2)) \
                + xp2.unsqueeze_(2).add(x2.unsqueeze_(1))

            xp.squeeze_(0)

        x.squeeze_(0)

        return sqd.squeeze(0)

    def kernel(self, x: Tensor, xp: Tensor = None) -> Tensor:

        hp = self.params

        x = x.view((-1, x.shape[-2], x.shape[-1]))

        hp = hp.view((-1, hp.shape[-1]))

        sig = hp[:, 0]
        sig_noise = hp[:, 1]
        ls = hp[:, 2:]
        eps = sig_noise.square()
        ls.unsqueeze_(1)
        xl = x.mul(ls)

        if xp is None:
            sqd = self.distance(xl)

            sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

            sqd.mul_(-1.0)
            sqd.exp_()
            sqd.mul_(sig[:, None, None].square())

            idt = tc.empty_like(sqd).copy_(tc.eye(sqd.shape[-1]))
            idt.mul_(eps[:, None, None])
            sqd.add_(idt)

        else:
            xp = xp.view((-1, xp.shape[-2], xp.shape[-1]))
            xpl = xp.mul(ls)

            sqd = self.distance(xl, xp=xpl)
            sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

            sqd.mul_(-1.0)
            sqd.exp_()
            sqd.mul_(sig[:, None, None].square())

        hp.squeeze_(0)

        sqd.squeeze_(0)

        return sqd

    def kernel_and_grad(self, x: Tensor) -> List[Tensor]:

        hp = self.params

        hp = hp.view((-1, hp.shape[-1]))

        krn = self.kernel(x)

        x = x.view((-1, x.shape[-2], x.shape[-1]))
        krn = krn.view((-1, krn.shape[-2], krn.shape[-1]))

        nc = x.shape[0]
        nhp = hp.shape[-1]
        n = krn.shape[-1]

        dkrn = tc.empty([nc, nhp, n, n])

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
        krn.squeeze_(0)
        dkrn.squeeze_(0)

        return [krn, dkrn]


# Docs here

Covar.kernel.__doc__ = """
Covariance kernel matrix for batched samples x and test samples xp.

Parameters:
    x: Tensor[..., n, dim]
        Batched Training samples.
    xp: Tensor[..., m, dim]
        Batched Test samples
    hp: Tensor[..., nhp]
        Batched Hyperparameter of the covariance,

Returns:
    krn_mat: Tensor[..., m, n]
                Covariance kernel matrix K(x,x')
"""


Covar.kernel_and_grad.__doc__ = """
Derivative of the covariance kernel wrt hyperparameters.
    dK(x,x) / d\theta_i

Parameters:
    x: Tensor[..., n, dim]
        Batched training samples
    hp: Tensor[..., nhp]
        Batched hyperparameters at which the derivative is taken.
        if None, the object internal Covar.hyper_parameter is used.

Returns:
    dkrn: Tensor[..., nhp, n, n]
          Batched matrix derivative wrt each hyperparameter.
"""


Squared_exponential.distance.__doc__ = """
Euclidean distance matrix between x and xp.

Parameters:
    x: Tensor[..., n, dim]
        Batched train samples
    xp: Tensor[..., m, dim]
        Batched test samples

N.B: Only one of x or xp allowed to be batched.

Returns:
    sqd: Tensor[..., m, n]
        Distance matrix between x and xp
        if xp = None then returns distance matrix of x.
"""
