from collections import deque

import torch as T


class History:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.feature_window: deque[T.Tensor] = deque(maxlen=max_length)
        self.done_window: deque[T.Tensor] = deque(maxlen=max_length)

    def push(self, feature: T.Tensor, done: T.Tensor) -> None:
        self.feature_window.append(feature)
        self.done_window.append(done)

    def _pad_window(self, window: deque[T.Tensor], default: T.Tensor) -> T.Tensor:
        """Pad the window to max_length with default values."""
        padded_window = (self.max_length - len(window)) * [default] + list(window)
        return T.stack(padded_window, dim=1)

    def peek(self) -> tuple[T.Tensor, T.Tensor]:
        return (
            self._pad_window(self.feature_window, self.feature_window[0].clone()),
            self._pad_window(self.done_window, T.zeros_like(self.done_window[0])),
        )

    def reset(self) -> None:
        self.window.clear()
