from rl_lib.training.settings.loss import (
    EntropySettings,
)


class EntropyDecay:
    def __init__(self, cfg: EntropySettings):
        self.coefficient_init = cfg.coefficient
        self.decay = cfg.decay
        self.coefficient_min = cfg.min_coefficient
        self._factor = 1.0

    @property
    def coefficient(self) -> float:
        return max(
            self.coefficient_init * self._factor,
            self.coefficient_min,
        )

    def step(self) -> None:
        self._factor *= self.decay
