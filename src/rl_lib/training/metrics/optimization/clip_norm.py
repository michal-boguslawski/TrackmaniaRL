from rl_lib.training.metrics.contexts import OptimizationContext


def calculate_clip_norms(optimization_context: OptimizationContext) -> dict[str, float]:
    return {"clip_norm": optimization_context.clip_norm.item()}
