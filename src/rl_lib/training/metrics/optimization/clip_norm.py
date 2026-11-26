import torch as T
from torch.nn.utils import clip_grad_norm_

from rl_lib.training.metrics.contexts import OptimizationContext
from rl_lib.training.utils import tensor_metrics_to_scalars


def calculate_clip_norms(optimization_context: OptimizationContext) -> dict[str, float]:
    optimizer = optimization_context.optimizer
    inf = float("inf")
    params = [p for pg in optimizer.param_groups for p in pg["params"]]
    metrics = {
        "clip_norm": clip_grad_norm_(params, inf),
        **{
            f"clip_norm/{pg['name']}": clip_grad_norm_(pg["params"], inf)
            for pg in optimizer.param_groups
        },
    }
    return tensor_metrics_to_scalars(metrics)
