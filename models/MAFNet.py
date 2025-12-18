import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
from .fnet_mobilenetv2 import FeatureNet_mbv2, DeconvLayer

import math
import gc
from .aggregation import Aggregation, LinformerFusion
import time

from fre_attention import *



class GWCostVolume(nn.Module):
    def __init__(self, num_groups=8):
        super(CostVolume, self).__init__()
        self.num_groups = num_groups

        self.conv = BasicConv(64, 32, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1)

    def forward(self, left, right, maxdisp):
        """
        left, right: [B, 64, H, W]
        return: cost volume [B, maxdisp, H, W]
        """
        left = self.desc(self.conv(left))    # [B, 32, H, W]
        right = self.desc(self.conv(right))  # [B, 32, H, W]

        B, C, H, W = left.shape
        Ng = self.num_groups
        assert C % Ng == 0,
        Cg = C // Ng

        # reshape to group-wise features
        left = left.view(B, Ng, Cg, H, W)
        right = right.view(B, Ng, Cg, H, W)

        cv = []

        for d in range(maxdisp):
            if d > 0:
                # group-wise correlation
                cost = (
                    left[:, :, :, :, d:] *
                    right[:, :, :, :, :-d]
                ).mean(dim=2)        
                cost = cost.mean(dim=1, keepdim=True)  
                cost = F.pad(cost, (d, 0, 0, 0))
            else:
                cost = (left * right).mean(dim=2)
                cost = cost.mean(dim=1, keepdim=True)

            cv.append(cost)

        return torch.cat(cv, dim=1)



    

class MAFNet(nn.Module):
    def __init__(self, args):
        super(MAFNet, self).__init__()
        self.fnet = FeatureNet_mbv2()
        self.cost_stem = BasicConv(48, 32, kernel_size=3, stride=1, padding=1)

        self.cost_agg0 = Aggregation(in_channels=32,
                                    left_att=True,
                                    blocks=[4, 6, 8],
                                    expanse_ratio=4,
                                    backbone_channels=[64, 64, 192])

        self.cost_agg1 = Aggregation(in_channels=32,
                                    left_att=True,
                                    blocks=[4, 6, 8],
                                    expanse_ratio=4,
                                    backbone_channels=[64, 64, 192])

        self.spa_att = FrequencyDomainAttention()

        
        self.stem_2 = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
            BasicConv(16, 16, kernel_size=3, stride=1, padding=1)
            )
        self.stem_4 = nn.Sequential(
            BasicConv(16, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
            )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*16, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = DeconvLayer(32, 16)
        self.spx_4 = nn.Sequential(
            BasicConv(64, 32, kernel_size=3, stride=1, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
            )

        self.build_cv = GWCostVolume()

        self.fusion = LinformerFusion(
            in_channels=48, 
            k=256, depth=1, heads=4
        )
        
    def upsample_disp(self, disp, mask, scale=4):
        """ Upsample disp field [H//4, W//4] -> [H, W] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(scale * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, scale*H, scale*W)

    def forward(self, left, right, max_disp=192):

        features_left = self.fnet(left)
        features_right = self.fnet(right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        corr = self.build_cv(features_left[0], features_right[0], max_disp//4)
        cv = self.cost_stem(corr)

        spa_att = self.spa_att(features_left)
        cv_0 = spa_att * cv
        cv_1 = (1. - spa_att) * cv
        cv_0 = self.cost_agg0(cv_0, features_left)
        cv_1 = self.cost_agg1(cv_1, features_left)
        cv = self.fusion(cv_0, cv_1)

        prob = F.softmax(cv, dim=1)
        disp = disparity_regression(prob, max_disp // 4) 

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_up = context_upsample(disp, spx_pred)
        
        
        if self.training:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up*4., disp_linear*4.]
        else:
            return disp_up*4.


if __name__ == '__main__':
    left = torch.randn(1, 3, 256, 512)
    right = torch.randn(1, 3, 256, 512)
    fnet = MAFNet(args=None)
    disp_pred = fnet(left, right)
    print("Training mode output shapes:")
    print(f"disp_up: {disp_pred[0].shape}")
    print(f"disp_linear: {disp_pred[1].shape}")
    
