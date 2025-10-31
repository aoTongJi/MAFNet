import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(FrequencyDomainAttention, self).__init__()
        
        self.conv0 = BasicConv(64, 32, kernel_size=3, stride=1, padding=1)  
        self.conv1 = BasicConv(64, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = BasicConv(192, 32, kernel_size=3, stride=1, padding=1) 
        self.freq_conv = nn.Sequential(
            BasicConv(32*3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 1)
        )
        
        self.attn = nn.Conv2d(32, 1, 5, padding=2, bias=False)
        
        self.high_freq_threshold = 0.1
        self.low_freq_threshold = 0.05

    def extract_frequency_features(self, x):
        B, C, H, W = x.shape
        
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        magnitude = torch.abs(x_fft_shift)
        center_h, center_w = H // 2, W // 2
        h_indices = torch.arange(H, device=x.device).view(1, -1, 1)
        w_indices = torch.arange(W, device=x.device).view(1, 1, -1)
        
        h_dist = (h_indices - center_h) ** 2
        w_dist = (w_indices - center_w) ** 2
        dist = torch.sqrt(h_dist + w_dist).squeeze(0)
        min_dim = min(H, W)
        high_freq_mask = (dist > min_dim * self.high_freq_threshold).float()
        low_freq_mask = (dist < min_dim * self.low_freq_threshold).float()

        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

        high_freq = magnitude * high_freq_mask
        low_freq = magnitude * low_freq_mask
        

        freq_diff = high_freq - low_freq

        freq_diff_ishift = torch.fft.ifftshift(freq_diff, dim=(-2, -1))
        freq_feature = torch.fft.ifft2(freq_diff_ishift, dim=(-2, -1))
        freq_feature = torch.abs(freq_feature)
        
        return freq_feature

    def forward(self, features_left):
        x4 = self.conv0(features_left[0]) 
        x8 = self.conv1(features_left[1]) 
        x16 = self.conv2(features_left[2])
        
        x8_4 = F.interpolate(x8, x4.shape[2:], mode='bilinear', align_corners=False)
        x16_4 = F.interpolate(x16, x4.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x4, x8_4, x16_4], dim=1)
        
        freq_features = self.extract_frequency_features(x)
        
        spatial_freq_features = self.freq_conv(x + freq_features)
        
        att = self.attn(spatial_freq_features)
        att = torch.sigmoid(att)
        
        return att

class ImprovedFrequencyDomainAttention(nn.Module):

    def __init__(self, init_low=0.06, init_high=0.18, tau=0.02,
                 in_chs_4=64, in_chs_8=64, in_chs_16=192, mid_ch=32):
        super().__init__()

        self.conv4 = BasicConv(in_chs_4,  mid_ch, kernel_size=3, stride=1, padding=1)
        self.conv8 = BasicConv(in_chs_8,  mid_ch, kernel_size=3, stride=1, padding=1)
        self.conv16= BasicConv(in_chs_16, mid_ch, kernel_size=3, stride=1, padding=1)

        self.pre_fuse = nn.Sequential(
            BasicConv(mid_ch * 3, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_ch, mid_ch, 1)
        )


        self.band_proj = nn.ModuleList([
            BasicConv(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            BasicConv(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            BasicConv(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0),
        ])

        self.gate_head = nn.Conv2d(mid_ch * 3, 3, kernel_size=1, stride=1, padding=0)

        self.post_fuse = BasicConv(mid_ch, 32, kernel_size=3, stride=1, padding=1)
        self.attn = nn.Conv2d(32, 1, kernel_size=5, padding=2, bias=False)

        self.tl_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(init_low)))
        self.th_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(init_high)))

        self.tau = tau

        self.register_buffer('_r_norm', torch.empty(0), persistent=False)

    @staticmethod
    def _inv_sigmoid(x, eps=1e-6):
        x = x.clamp(eps, 1-eps)
        return math.log(x/(1-x))

    def _get_thresholds(self):
        tl = torch.sigmoid(self.tl_raw) * 0.5
        th = torch.sigmoid(self.th_raw) * 0.5
        t_low, t_high = torch.minimum(tl, th), torch.maximum(tl, th)
        gap = 0.01
        t_high = torch.clamp(t_high, min=(t_low + gap), max=0.5)
        return t_low, t_high

    def _get_radius_grid(self, H, W, device):
        if self._r_norm.numel() and self._r_norm.shape[-2:] == (H, W):
            return self._r_norm
        yy = torch.arange(H, device=device).float().view(H, 1)
        xx = torch.arange(W, device=device).float().view(1, W)
        cy, cx = (H-1)/2.0, (W-1)/2.0
        r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)             
        r_max = torch.sqrt((cy)**2 + (cx)**2)                   
        r_norm = (r / (2.0 * r_max)).clamp(0, 0.5)              
        self._r_norm = r_norm
        return r_norm

    def _soft_band_masks(self, H, W, device):

        t_low, t_high = self._get_thresholds()
        r = self._get_radius_grid(H, W, device)                  

        r = r.unsqueeze(0).unsqueeze(0)

        sig = torch.sigmoid
        tau = self.tau

        low_mask  = sig((t_low  - r) / tau)                       
        high_mask = sig((r - t_high) / tau)                       


        mid_left  = sig((r - t_low) / tau)
        mid_right = sig((t_high - r) / tau)
        mid_mask  = mid_left * mid_right

        denom = (low_mask + mid_mask + high_mask).clamp_min(1e-6)
        low_mask  = low_mask  / denom
        mid_mask  = mid_mask  / denom
        high_mask = high_mask / denom
        return low_mask, mid_mask, high_mask  

    def _fft_bands(self, x):
        B, C, H, W = x.shape

        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))                

        low_m, mid_m, high_m = self._soft_band_masks(H, W, x.device)  

        low_m  = low_m.expand(B, C, H, W)
        mid_m  = mid_m.expand(B, C, H, W)
        high_m = high_m.expand(B, C, H, W)


        X_low  = X * low_m
        X_mid  = X * mid_m
        X_high = X * high_m

        def ifft_mag(Xb):
            Xb = torch.fft.ifftshift(Xb, dim=(-2, -1))
            xb = torch.fft.ifft2(Xb, dim=(-2, -1))               
            return torch.abs(xb)                                 

        low  = ifft_mag(X_low)   
        mid  = ifft_mag(X_mid)
        high = ifft_mag(X_high)

        out = torch.stack([low, mid, high], dim=1)  
        return out

    def forward(self, features_left):


        x4, x8, x16 = features_left

        x4_ = self.conv4(x4)
        x8_ = F.interpolate(self.conv8(x8), size=x4_.shape[2:], mode='bilinear', align_corners=False)
        x16_= F.interpolate(self.conv16(x16), size=x4_.shape[2:], mode='bilinear', align_corners=False)


        x = torch.cat([x4_, x8_, x16_], dim=1)     
        x = self.pre_fuse(x)                       


        bands = self._fft_bands(x)                


        proj = []
        for i in range(3):
            bi = self.band_proj[i](bands[:, i, :, :, :])  
            proj.append(bi)
        proj_cat = torch.cat(proj, dim=1)                 

        gate_logits = self.gate_head(proj_cat)            
        gate = F.softmax(gate_logits, dim=1)              


        fused = (gate[:, 0:1] * proj[0] +
                 gate[:, 1:2] * proj[1] +
                 gate[:, 2:3] * proj[2])               

        fused = self.post_fuse(fused)                  
        att = torch.sigmoid(self.attn(fused))          
        return att