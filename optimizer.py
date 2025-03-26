from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW optimizer with decoupled weight decay regularization.
    Reference:
    - "Adam: A Method for Stochastic Optimization" by Kingma & Ba (2014)
    - "Decoupled Weight Decay Regularization" by Loshchilov & Hutter (2017)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        """
        Initialize the AdamW optimizer.

        Args:
            params (Iterable[torch.nn.parameter.Parameter]): Model parameters to optimize.
            lr (float): Learning rate. Default: 1e-3.
            betas (Tuple[float, float]): Coefficients for computing running averages of gradient and its square. Default: (0.9, 0.999).
            eps (float): Term added to denominator for numerical stability. Default: 1e-6.
            weight_decay (float): Weight decay coefficient. Default: 0.0.
            correct_bias (bool): Whether to correct bias in moment estimates. Default: True.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value returned by the closure (if provided).
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Update step count
                state["step"] += 1
                step = state["step"]

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    corrected_exp_avg = exp_avg / bias_correction1
                    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                else:
                    corrected_exp_avg = exp_avg
                    corrected_exp_avg_sq = exp_avg_sq

                # Compute step size
                denom = corrected_exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]

                # Update parameters
                p.data.addcdiv_(corrected_exp_avg, denom, value=-step_size)

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss