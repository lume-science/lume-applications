import torch
from typing import Callable, Dict

def generalized_taylor(model, input_names: Dict[str, torch.Tensor], output_names: Dict[str, torch.Tensor], g: Callable[[torch.Tensor], torch.Tensor]):
    """Compute the generalized Jacobian of the model outputs with respect to the inputs after taking the output through some g function."""
    init = model.get(input_names)
    x0 = torch.stack([init[k] for k in input_names]).detach()
    out0 = model.get(output_names)
    y0 = torch.stack([out0[k] for k in output_names]).detach()

    def forward(x):
        model.set({k: x[i] for i, k in enumerate(input_names)})
        vals = model.get(output_names)
        y = torch.stack([vals[k] for k in output_names])
        return g(y)

    J = torch.autograd.functional.jacobian(forward, x0)
    H = torch.autograd.functional.hessian(forward, x0)
    model.set({k: x0[i] for i, k in enumerate(input_names)})  # leave the model at x0, without the autograd graph
    return J.detach(), H.detach(), x0, y0

class GenerizalizedTaylorPrior(torch.nn.Module):

    def __init__(self, jacobian, hessian, z0, f0):
        super().__init__()
        self.register_buffer("J", jacobian.to(torch.float64))
        self.register_buffer("H", hessian.to(torch.float64))
        self.register_buffer("z0", z0.to(torch.float64))
        self.register_buffer("f0", f0.to(torch.float64))
    def forward(self, x):
        # x: (..., z) input tensor
        delta = x - self.z0
        linear_term = delta @ self.J.T
        quadratic_term = 0.5 * torch.einsum('...i,ij,...j->...', delta, self.H, delta)
        return self.f0 + linear_term + quadratic_term

