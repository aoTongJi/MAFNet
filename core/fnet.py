import torch
import torch.nn as nn
import os

# The exact pretrained artifact is already cached on the experiment host.
# Offline mode prevents a transient network failure from changing launch
# behavior while still letting timm load the cached ImageNet checkpoint.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import timm
from .submodule import BasicConv

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = BasicConv(in_channels, out_channels, deconv=True, bn=True, relu=True, kernel_size=3, stride=2, padding=1, output_padding=1,)
        self.concat = BasicConv(out_channels*2, out_channels*2, bn=True, relu=True, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = self.deconv(x)
        xy = torch.cat([x, y], 1)
        out = self.concat(xy)
        return out

class FeatureNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # The stereo network is trained from scratch, while both compared models
        # may share the same ImageNet-initialized MobileNetV2 feature extractor.
        model = timm.create_model(
            'mobilenetv2_100', pretrained=pretrained, features_only=True)
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # timm>=0.9 wraps the stem activation inside BatchNormAct2d; older
        # versions expose it separately as ``act1``.
        self.act1 = getattr(model, 'act1', nn.Identity())
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.deconv32_16 = DeconvLayer(chans[4], chans[3])
        self.deconv16_8 = DeconvLayer(chans[3]*2, chans[2])
        self.deconv8_4 = DeconvLayer(chans[2]*2, chans[1])
        self.conv4 = BasicConv(chans[1]*2, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)

        return [x4, x8, x16]





    
