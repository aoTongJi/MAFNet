import torch
import torch.nn as nn
import timm
from .submodule import BasicConv
from mmpretrain import get_model

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


class FeatureNet_mbv2(nn.Module):
    def __init__(self):
        super().__init__()

        model = get_model(
            'mobilenet-v2_8xb32_in1k',
            pretrained=True
        )
        self.backbone = model.backbone

        self.stem   = self.backbone.conv1  
        self.layer1 = self.backbone.layer1  
        self.layer2 = self.backbone.layer2   
        self.layer3 = self.backbone.layer3   
        self.layer4 = self.backbone.layer4    
        self.layer5 = self.backbone.layer5   

        # -------- Decoder --------
        self.deconv32_16 = DeconvLayer(320, 96)
        self.deconv16_8  = DeconvLayer(96 * 2, 32)
        self.deconv8_4   = DeconvLayer(32 * 2, 24)

        self.conv4  = BasicConv(24 * 2, 32, kernel_size=3, padding=1)
        self.conv8  = BasicConv(32, 64, kernel_size=3, padding=1)
        self.conv16 = BasicConv(96, 192, kernel_size=3, padding=1)

    def forward(self, x):
        x2  = self.stem(x)           # 1/2
        x2  = self.layer1(x2)        # 1/2
        x4  = self.layer2(x2)        # 1/4
        x8  = self.layer3(x4)        # 1/8
        x16 = self.layer4(x8)        # 1/16
        x32 = self.layer5(x16)       # 1/32

        x16_up = self.deconv32_16(x32, x16)
        x8_up  = self.deconv16_8(x16_up, x8)
        x4_up  = self.deconv8_4(x8_up, x4)

        x4_out  = self.conv4(x4_up)
        x8_out  = self.conv8(x8)
        x16_out = self.conv16(x16)

        return [x4_out, x8_out, x16_out]




if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 512)
    net = FeatureNet_mbv2()
    feats = net(x)
    print(feats[0].shape, feats[1].shape, feats[2].shape)




    
