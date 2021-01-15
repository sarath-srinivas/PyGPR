import torch as tc
from torch import Tensor
from .loss import Loss


def update_model(loss: Loss, y: Tensor, eps: float) -> None:
    """
     Update GPR model hyperparameters with new sample y
      where y - y_old ~ O(eps)
    """

    loss.model.y = tc.clone(y)
    loss.model.need_upd = True

    old_params = tc.clone(loss.model.params).numpy()

    f0, J = loss.loss_and_grad(old_params)

    fp = loss.loss(old_params + eps * J)
    fm = loss.loss(old_params - eps * J)

    C1 = (fp + fm) / (2.0 * eps)
    C2 = (fp + fm - 2 * f0) / (2.0 * eps ** 2)

    gamma = -0.5 * (C1 / C2)

    new_params = old_params - gamma * J

    loss.model.set_params(tc.from_numpy(new_params))

    return None
