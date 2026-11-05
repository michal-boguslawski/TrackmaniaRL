import torch as T
from torch import nn

from rl_lib.networks.cnn import CNN
from rl_lib.networks.config import EncoderConfig
from rl_lib.networks.temporal import TemporalCNN1D


class Encoder(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        stack_size: int,
        config: EncoderConfig,
        cnn_autocast_dtype: T.dtype | None = None,
    ):
        super().__init__()
        self.cfg = config
        self.stack_size = stack_size
        self.observation_dim = observation_dim
        self.cnn_autocast_dtype = cnn_autocast_dtype

        self.cnn = CNN(observation_dim, self.cfg.cnn)

        self.sequence_encoder = TemporalCNN1D(
            stack_size, in_dim=self.cnn.out_dim, config=self.cfg.temporal
        )
        self.register_buffer(
            "_divisor", T.tensor(self.cfg.observation_divisor), persistent=False
        )
        self.register_buffer(
            "_offset", T.tensor(self.cfg.observation_offset), persistent=False
        )

    def _normalize(self, frames_nhwc_uint8: T.Tensor) -> T.Tensor:
        x = frames_nhwc_uint8.to(self._divisor.device, T.float32)
        x = T.addcdiv(
            self._offset, x, self._divisor
        )  # same rounding as x / divisor + offset
        return x.permute(0, 3, 1, 2)

    def frame_features(
        self, frames: T.Tensor
    ) -> T.Tensor:  # (N,H,W,C) uint8 -> (N,D), CNN only
        normalized_frames = self._normalize(frames)  # (N,C,H,W)
        autocast_dtype = getattr(self, "cnn_autocast_dtype", None)
        if autocast_dtype is None:
            return self.cnn(normalized_frames).float()
        with T.autocast(device_type="cuda", dtype=autocast_dtype):
            features = self.cnn(normalized_frames)
        return features.float()

    def aggregate(
        self, feats: T.Tensor, mask: T.Tensor = None
    ) -> T.Tensor:  # (B,T,D) -> (B,D), temporal only
        if mask is not None:
            feats.masked_fill_(mask.unsqueeze(-1), 0.0)
        return self.sequence_encoder(feats)

    def forward(
        self, frames: T.Tensor, mask: T.Tensor = None
    ) -> T.Tensor:  # (B,T,H,W,C) -> (B,D), the training entry point
        B, T, H, W, C = frames.shape
        frames = frames.view(B * T, H, W, C)
        feats = self.frame_features(frames)  # (B*T,D)
        feats = feats.view(B, T, -1)  # (B,T,D)
        return self.aggregate(feats, mask)  # (B,D)

    @property
    def out_dim(self) -> int:
        return self.cnn.out_dim
