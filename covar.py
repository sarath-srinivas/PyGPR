import torch as tc
from torch import Tensor
from typing import List, Sequence, Protocol

tc.set_default_tensor_type(tc.DoubleTensor)
tc.set_printoptions(precision=7, sci_mode=True)


class Covar(Protocol):
    """
     Protocol for covariance kernels for Gaussian process
     regression.
    """

    def get_params_shape(self, x: Tensor) -> List[int]:
        ...

    def init_params(self, x: Tensor) -> Tensor:
        ...

    def kernel(self, params: Tensor, x: Tensor, xp: Tensor = None) -> Tensor:
        ...

    def kernel_and_grad(self, params: Tensor, x: Tensor) -> List[Tensor]:
        ...


class Compose:
    """
     Class for composing covariance kernels to form new one.
    """

    def __init__(self, covars: Sequence[Covar]) -> None:
        self.covars = covars

    def get_params_shape(self, x: Tensor) -> List[int]:

        nparams = sum([covar.get_params_shape(x)[-1] for covar in self.covars])
        shape = list(x.shape)
        shape[-1] = nparams
        shape.pop(-2)

        return shape

    def init_params(self, x: Tensor) -> Tensor:

        params = [covar.init_params(x) for covar in self.covars]
        return tc.cat(params, dim=-1)

    def kernel(self, hp: Tensor, x: Tensor, xp: Tensor = None) -> Tensor:

        assert hp.shape[-1] == self.get_params_shape(x)[-1]

        chunks = [covar.get_params_shape(x)[-1] for covar in self.covars]
        params = hp.split(chunks, dim=-1)

        krn = self.covars[0].kernel(params[0], x, xp)

        for i in range(1, len(self.covars)):
            krn.add_(self.covars[i].kernel(params[i], x, xp))

        return krn

    def kernel_and_grad(self, hp: Tensor, x: Tensor) -> List[Tensor]:

        assert hp.shape[-1] == self.get_params_shape(x)[-1]

        chunks = [covar.get_params_shape(x)[-1] for covar in self.covars]
        params = hp.split(chunks, dim=-1)

        dkrns = []
        krn, dkrn1 = self.covars[0].kernel_and_grad(params[0], x)
        dkrns.append(dkrn1)

        for i in range(1, len(self.covars)):
            krn.add_(self.covars[i].kernel_and_grad(params[i], x)[0])
            dkrns.append(self.covars[i].kernel_and_grad(params[i], x)[1])

        dkrn = tc.cat(dkrns, dim=-3)

        return [krn, dkrn]


class Squared_exponential:
    """
     Squared exponential covariance K(x,x') = sig_y * exp(-|(x-x').ls|^2)
    """

    def get_params_shape(self, x: Tensor) -> List[int]:

        shape = list(x.shape)
        shape[-1] = x.shape[-1] + 1
        shape.pop(-2)
        return shape

    def init_params(self, x: Tensor) -> Tensor:

        shape = self.get_params_shape(x)
        params = tc.ones(shape)
        return params

    def distance(self, x: Tensor, xp: Tensor = None) -> Tensor:

        x = x.view((-1, x.shape[-2], x.shape[-1]))

        x2 = tc.sum(x.square(), 2)

        if xp is None:

            sqd = -2.0 * tc.matmul(x, x.transpose(1, 2)) + x2.unsqueeze(2).add(
                x2.unsqueeze(1)
            )

        else:
            xp = xp.view((-1, xp.shape[-2], xp.shape[-1]))

            xp2 = tc.sum(xp.square(), 2)

            sqd = -2.0 * tc.matmul(xp, x.transpose(1, 2)) + xp2.unsqueeze_(
                2
            ).add(x2.unsqueeze_(1))

            xp.squeeze_(0)

        x.squeeze_(0)

        return sqd.squeeze(0)

    def kernel(self, hp: Tensor, x: Tensor, xp: Tensor = None) -> Tensor:

        assert hp.shape[-1] == self.get_params_shape(x)[-1]

        x = x.view((-1, x.shape[-2], x.shape[-1]))

        hp = hp.view((-1, hp.shape[-1]))

        sig = hp[:, 0]
        ls = hp[:, 1:]
        ls.unsqueeze_(1)
        xl = x.mul(ls)

        if xp is None:
            sqd = self.distance(xl)

            sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

            sqd.mul_(-1.0)
            sqd.exp_()
            sqd.mul_(sig[:, None, None].square())

        else:
            xp = xp.view((-1, xp.shape[-2], xp.shape[-1]))
            xpl = xp.mul(ls)

            sqd = self.distance(xl, xp=xpl)
            sqd = sqd.view((-1, sqd.shape[-2], sqd.shape[-1]))

            sqd.mul_(-1.0)
            sqd.exp_()
            sqd.mul_(sig[:, None, None].square())

        hp.squeeze_(0)
        x.squeeze_(0)

        sqd.squeeze_(0)

        return sqd

    def kernel_and_grad(self, hp: Tensor, x: Tensor) -> List[Tensor]:

        assert hp.shape[-1] == self.get_params_shape(x)[-1]

        krn = self.kernel(hp, x)

        hp = hp.view((-1, hp.shape[-1]))

        x = x.view((-1, x.shape[-2], x.shape[-1]))
        krn = krn.view((-1, krn.shape[-2], krn.shape[-1]))

        nc = x.shape[0]
        nhp = hp.shape[-1]
        n = krn.shape[-1]

        dkrn = tc.empty([nc, nhp, n, n])

        sig = hp[:, 0]
        ls = hp[:, 1:]

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
        krn.squeeze_(0)
        dkrn.squeeze_(0)

        return [krn, dkrn]


class White_noise:
    """
     Gaussian noise covariance kernel
    """

    def get_params_shape(self, x: Tensor) -> List[int]:

        shape = list(x.shape)
        shape[-1] = 1
        shape.pop(-2)
        return shape

    def init_params(self, x: Tensor) -> Tensor:

        shape = self.get_params_shape(x)
        params = 1e-4 * tc.ones(shape)
        return params

    def kernel(self, hp: Tensor, x: Tensor, xp: Tensor = None) -> Tensor:

        if xp is None:
            hpb = hp.view(-1, hp.shape[-1])

            nc = hpb.shape[0]
            n = x.shape[-2]

            sig_n = hpb[:, 0]

            krn = tc.empty([nc, n, n]).copy_(tc.eye(n))
            krn.mul_(sig_n[:, None, None].square())

            krn.squeeze_(0)

        else:
            krn = tc.tensor(0)

        return krn

    def kernel_and_grad(self, hp: Tensor, x: Tensor) -> List[Tensor]:

        hpb = hp.view(-1, hp.shape[-1])

        krn = self.kernel(hp, x)
        krnb = krn.view(-1, krn.shape[-2], krn.shape[-1])

        nc = hpb.shape[0]
        n = krn.shape[-1]
        nhp = 1

        sig_n = hpb[:, 0]

        dkrn = tc.empty([nc, nhp, n, n])

        dkrn[:, 0, :, :] = krnb.mul(
            sig_n[:, None, None].reciprocal().mul_(2.0)
        )

        krn.squeeze_(0)
        dkrn.squeeze_(0)

        return [krn, dkrn]


# Docs here

Covar.get_params_shape.__doc__ = """
Returns shape of parameters of the covariance kernel from the \
shape of the sample tensor x.

Parameters
----------
x: Tensor[nb,n,dim] or Tensor[n,dim]
    The samples tensor.

Returns
-------
List[nb, np] or List[np]
    The shape information of the parameters of the covariance kernel.
"""

Covar.kernel.__doc__ = """
Covariance kernel matrix for batched samples x and test samples xp.

Parameters
----------
params: Tensor[shape]
    Kernel hyperparameters of shape given by :obj:get_params_shape(x)
x: Tensor[..., n, dim]
    Batched Training samples.
xp: Tensor[..., m, dim]
    Batched Test samples
hp: Tensor[..., nhp]
    Batched Hyperparameter of the covariance,

Returns
-------
Tensor[..., m, n]
    Covariance kernel matrix K(x,x')
"""

Covar.kernel_and_grad.__doc__ = """
Derivative of the covariance kernel wrt hyperparameters.
:math:`\frac{dK(x,x)}{d\theta_i}`

Parameters
----------
params: Tensor[shape]
    Kernel hyperparameters of shape given by :obj:get_params_shape(x)
x: Tensor[..., n, dim]
    Batched training samples
hp: Tensor[..., nhp]
    Batched hyperparameters at which the derivative is taken.
    if None, the object internal Covar.hyper_parameter is used.

Returns
-------
Tensor[..., nhp, n, n]
    Batched matrix derivative wrt each hyperparameter.
"""

Squared_exponential.distance.__doc__ = """
Euclidean distance matrix between x and xp.

Parameters
----------
x: Tensor[..., n, dim]
    Batched train samples
xp: Tensor[..., m, dim]
    Batched test samples

N.B: Only one of x or xp allowed to be batched.

Returns
-------
Tensor[..., m, n]
    Distance matrix between x and xp
    if xp = None then returns distance matrix of x.
"""

Compose.__init__.__doc__ = """

Returns a Covar object composed form covars.

Parameters
----------
covars: List[Covar]
    List of Covar kernel classes to be added

Returns
-------
Covar
    Covar kernel object.
"""
