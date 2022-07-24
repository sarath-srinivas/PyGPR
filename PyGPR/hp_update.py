import torch as tc
from torch import Tensor
from .loss import Loss


def get_learn_rate(current_param: Tensor, loss_new: Loss, eps: float) -> float:
    """
     Update GPR model hyperparameters with new sample y
      where y - y_old ~ O(eps)
    """

    old_params = tc.clone(current_param).numpy()

    f0, J = loss_new.loss_and_grad(old_params)

    fp = loss_new.loss(old_params - eps * J)
    fm = loss_new.loss(old_params + eps * J)

    C1 = (fp - fm) / (2.0 * eps)
    C2 = (fp + fm - 2 * f0) / (2.0 * eps ** 2)

    gamma = -0.5 * (C1 / C2)

    # new_params = old_params - gamma * J

    # loss.model.set_params(tc.from_numpy(new_params))

    return gamma
