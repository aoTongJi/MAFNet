import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *


from linformer import Linformer


class LinformerFusion(nn.Module):
    def __init__(self, in_channels=48, k=256, depth=1, heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.depth = depth
        self.heads = heads
        self.project = nn.Linear(in_channels * 2, in_channels)

    def forward(self, detail, smooth):
        B, C, H, W = detail.shape
        seq_len = H * W
        
        linformer = Linformer(
            dim=C * 2,  # detail 和 smooth 拼接
            seq_len=seq_len,
            depth=self.depth,
            heads=self.heads,
            k=self.k
        )
        device = detail.device
        linformer = linformer.to(device)
        
        x = torch.cat([detail, smooth], dim=1)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, 2 * C)
        x = linformer(x)  # [B, N, 2C]
        x = self.project(x)    # [B, N, C]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x
