import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
from .fnet import FeatureNet, DeconvLayer
import math
import gc
from .aggregation import Aggregation
import time



class CostVolume(nn.Module):
    def __init__(self):
        super(CostVolume, self).__init__()
        self.conv = BasicConv(64, 32, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1)
        self.reduce_mean = nn.Conv2d(32, 1, kernel_size=1,stride=1,padding=0,bias=False)

        self.reduce_mean.weight.data.fill_(1.0 / 32.0)
        self.reduce_mean.weight.requires_grad = False


    def forward(self, left, right, maxdisp):
        left = self.desc(self.conv(left))
        right = self.desc(self.conv(right))
        cv = []
        for i in range(maxdisp):
            if i > 0:
                cost = self.reduce_mean(left[:,:,:,i:] * right[:,:,:,:-i])
                cost = F.pad(cost, (i, 0, 0, 0))
                cv.append(cost)

            else:
                cost = self.reduce_mean(left * right)
                cv.append(cost)

        return torch.cat(cv, dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv0 = BasicConv(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = BasicConv(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = BasicConv(192, 32, kernel_size=3, stride=1, padding=1)
        
        self.lastconv = nn.Sequential(
            BasicConv(32*3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 1))

        self.attn = nn.Conv2d(32, 1, 5, padding=2, bias=False)

    def forward(self, features_left):
        x4 = self.conv0(features_left[0])
        x8 = self.conv1(features_left[1])
        x16 = self.conv2(features_left[2])
        x8_4 = F.interpolate(x8, x4.shape[2:], mode='bilinear', align_corners=False)
        x16_4 = F.interpolate(x16, x4.shape[2:], mode='bilinear', align_corners=False)
        x = self.lastconv(torch.cat([x4, x8_4, x16_4], dim=1))
        att = self.attn(x)
        att = torch.sigmoid(att)
        return att
    

class BANet(nn.Module):
    def __init__(self, args):
        super(BANet, self).__init__()
        self.fnet = FeatureNet(pretrained=True)
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

        self.spa_att = SpatialAttention()
        
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

        self.build_cv = CostVolume()
        
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
        cv = spa_att * cv_0 + (1. - spa_att) * cv_1

        prob = F.softmax(cv, dim=1)
        disp = disparity_regression(prob, max_disp // 4)  # [B, 1, H//4, W//4]

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_up = context_upsample(disp, spx_pred) # [B, 1, H, W]
        
        
        if self.training:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up*4., disp_linear*4.]
        else:
            return disp_up*4.
