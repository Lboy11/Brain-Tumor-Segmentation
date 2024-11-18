import torch
import torch.nn as nn
from modeling.ensemble.unet3d_parallel import UNet as UNetPara


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True, **kwargs):
        super().__init__()

        # 初始化参数
        self.in_channels = in_channels
        self.output = output
        self.feature = feature
        self.modality_specific_norm = modality_specific_norm
        self.width_ratio = width_ratio
        self.sharing = sharing
        self.module = UNetPara(1, out_channels, num_modalities=in_channels, parallel=True,
                            exchange=exchange, feature=feature, width_multiplier=width_ratio)


    def forward(self, x, channel=[], weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]
        out, df = self.module(x)

        if self.training:
            return out, df

        out = torch.stack(out, dim=0)
        preserved = list(range(self.in_channels))
        for c in channel:
            preserved.remove(c)

        return torch.mean(out[preserved], dim=0)



