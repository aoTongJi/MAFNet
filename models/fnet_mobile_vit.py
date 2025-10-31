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


class FeatureNet_vit(nn.Module):
    def __init__(self):
        super().__init__()
        
        model = get_model('mobilevit-small_3rdparty_in1k', pretrained=True)
        self.backbone = model.backbone
        
        chans = [32, 64, 96, 128, 160]

        self.stem = self.backbone.stem
        self.layer0 = self.backbone.layers[0]  
        self.layer1 = self.backbone.layers[1]  
        self.layer2 = self.backbone.layers[2]  
        self.layer3 = self.backbone.layers[3]  
        self.layer4 = self.backbone.layers[4]  

        self.deconv32_16 = DeconvLayer(chans[4], chans[3])        
        self.deconv16_8 = DeconvLayer(chans[3]*2, chans[2])       
        self.deconv8_4 = DeconvLayer(chans[2]*2, chans[1])        
        self.conv4 = BasicConv(chans[1]*2, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv8 = BasicConv(chans[2], 64, kernel_size=3, stride=1, padding=1)  
        self.conv16 = BasicConv(chans[3], 192, kernel_size=3, stride=1, padding=1) 

    def forward(self, x):
        x1 = self.stem(x)                   
        x2 = self.layer0(x1)                
        x4 = self.layer1(x2)                
        x8 = self.layer2(x4)                
        x16 = self.layer3(x8)               
        x32 = self.layer4(x16)              
        x16_up = self.deconv32_16(x32, x16)
        x8_up = self.deconv16_8(x16_up, x8)
        x4_up = self.deconv8_4(x8_up, x4)
        x4_out = self.conv4(x4_up)
        x8_out = self.conv8(x8)      
        x16_out = self.conv16(x16)   

        return [x4_out, x8_out, x16_out]  # [1/4, 1/8, 1/16]



if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 512)
    fnet = FeatureNet_vit()
    output = fnet(x)
    print(output[0].shape, output[1].shape, output[2].shape)




    