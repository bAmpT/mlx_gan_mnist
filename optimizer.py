from typing import List

import mlx.core as mx
from mlx.optimizers import Optimizer, OptimizerState


class AdamW(Optimizer):
    r"""Implementation of the AdamW optimizer [1].

    [1]: https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self, learning_rate: float = 0.001, betas: List[float] = [0.9, 0.999], eps: float = 1e-8, weight_decay: float = 0.01
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the Adam parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate
        b1, b2 = self.betas
        eps = self.eps
        weight_decay = self.weight_decay

        m = state.get("m", gradient)
        v = state.get("v", mx.square(gradient))
        t = state.get("t", mx.array([1.0]))
        m = b1 * m + (1.0 - b1) * gradient
        v = b2 * v + (1.0 - b2) * mx.square(gradient)
        mhat = m / (1.0 - b1 ** t.item())
        vhat = v / (1.0 - b2 ** t.item())

        state["m"] = m
        state["v"] = v
        state["t"] = t + 1

        return parameter - lr * (mhat / (mx.sqrt(vhat) + eps) + weight_decay * parameter)
        