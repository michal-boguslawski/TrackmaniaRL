import torch as T
from torch.nn.utils import clip_grad_norm_

from rl_lib.training.metrics.contexts import OptimizationContext
from rl_lib.training.utils import tensor_metrics_to_scalars


def calculate_clip_norms(optimization_context: OptimizationContext) -> dict[str, float]:
    groups = optimization_context.optimizer.param_groups

    group_norms = []
    for pg in groups:
        grads = [p.grad for p in pg["params"] if p.grad is not None]
        norms = T._foreach_norm(grads)  # one fused kernel launch per dtype/device
        group_norms.append(T.linalg.vector_norm(T.stack(norms)))
    group_norms = T.stack(group_norms)

    metrics = {
        "clip_norm": T.linalg.vector_norm(group_norms),
        **{f"clip_norm/{pg['name']}": n for pg, n in zip(groups, group_norms)},
    }
    return tensor_metrics_to_scalars(metrics)
