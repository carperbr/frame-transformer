import torch
from torch import log_softmax, nn
import torch.nn.functional as F
import math

from lib.frame_transformer_common import FrameNorm, FrameTransformerEncoder

class FrameTransformer(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FrameTransformer, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])

        self.out_norm = FrameNorm(self.max_bin, channels + num_transformer_blocks)
        self.out = nn.Linear(channels + num_transformer_blocks, 2, bias=bias)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        return F.pad(
            input=torch.sigmoid_(self.out(self.out_norm(x).transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )

class FrameTransformerDiscriminator(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FrameTransformerDiscriminator, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])
        self.out_norm = FrameNorm(self.max_bin, channels + num_transformer_blocks)
        self.out_channels = nn.Linear(channels + num_transformer_blocks, 1)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        return torch.mean(self.out_channels(self.out_norm(x).transpose(1,3)).transpose(1,3), dim=2, keepdim=True) 