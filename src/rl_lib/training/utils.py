from torch.optim import Adam, AdamW, Optimizer


def build_optimizer(params: list[dict], optimizer_name: str, eps: float) -> Optimizer:
    opt_cls = Adam if optimizer_name == "Adam" else AdamW
    return opt_cls(params, eps=eps)
