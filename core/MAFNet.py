"""MAFNet implemented on top of the BANet-2D feature and upsampling paths.

The two paper modules are implemented explicitly:
  * AFFA: adaptive RFFT high/low-frequency filtering and spatial gating.
  * AAHF: variable-resolution Linformer-style low-rank attention fusion.

No stereo-matching checkpoint is loaded.  As allowed by the experiment
protocol, BANet and MAFNet share the same ImageNet-pretrained MobileNetV2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BANet import CostVolume
from .aggregation import Aggregation
from .fnet import DeconvLayer, FeatureNet
from .submodule import BasicConv, context_upsample, disparity_regression


def _inverse_sigmoid(value):
    value = float(value)
    return math.log(value / (1.0 - value))


class AdaptiveFrequencyDomainFilteringAttention(nn.Module):
    """Adaptive Frequency-Domain Filtering Attention (AFFA).

    RFFT stores the horizontal non-negative frequencies only.  The radial grid
    therefore uses ``fftfreq`` vertically and ``rfftfreq`` horizontally, so DC
    is correctly located at (0, 0) without an incorrect full-spectrum shift.
    """

    def __init__(self, feature_channels=(64, 64, 192), hidden_dim=32,
                 temperature=0.08, initial_threshold=0.30):
        super().__init__()
        self.temperature = float(temperature)
        self.projections = nn.ModuleList([
            BasicConv(ch, hidden_dim, kernel_size=3, stride=1, padding=1)
            for ch in feature_channels
        ])
        self.multiscale_fusion = nn.Sequential(
            BasicConv(hidden_dim * len(feature_channels), hidden_dim,
                      kernel_size=3, stride=1, padding=1),
            BasicConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )

        # Sigmoid parameterization keeps both thresholds in the normalized
        # radial-frequency interval [0, 1].  Equal initialization makes the
        # two masks complementary at step zero: M_low + M_high = 1.
        threshold_logit = _inverse_sigmoid(initial_threshold)
        self.low_threshold_logit = nn.Parameter(torch.tensor(threshold_logit))
        self.high_threshold_logit = nn.Parameter(torch.tensor(threshold_logit))

        self.spatial_gate = nn.Conv2d(hidden_dim * 2, 2, kernel_size=1)
        self.attention_head = nn.Sequential(
            BasicConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    @staticmethod
    def _normalized_radius(height, width, device):
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.rfftfreq(width, device=device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        # Nyquist corner radius is sqrt(0.5^2 + 0.5^2).
        return radius / math.sqrt(0.5)

    def _frequency_split(self, x):
        height, width = x.shape[-2:]
        original_dtype = x.dtype

        # CUDA FFT on arbitrary spatial sizes is most robust in fp32.  The
        # cast remains differentiable and the result is returned to AMP dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            spectrum = torch.fft.rfft2(x_float, dim=(-2, -1), norm='ortho')
            radius = self._normalized_radius(height, width, x.device)
            radius = radius.view(1, 1, height, width // 2 + 1)
            tau_low = torch.sigmoid(self.low_threshold_logit)
            tau_high = torch.sigmoid(self.high_threshold_logit)
            low_mask = torch.sigmoid((tau_low - radius) / self.temperature)
            high_mask = torch.sigmoid((radius - tau_high) / self.temperature)
            low = torch.fft.irfft2(
                spectrum * low_mask, s=(height, width), dim=(-2, -1),
                norm='ortho')
            high = torch.fft.irfft2(
                spectrum * high_mask, s=(height, width), dim=(-2, -1),
                norm='ortho')

        return low.to(original_dtype), high.to(original_dtype)

    def forward(self, multiscale_features):
        target_size = multiscale_features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, multiscale_features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode='bilinear',
                    align_corners=False)
            projected.append(feature)

        x = self.multiscale_fusion(torch.cat(projected, dim=1))
        x_low, x_high = self._frequency_split(x)
        gates = torch.softmax(
            self.spatial_gate(torch.cat([x_low, x_high], dim=1)), dim=1)
        fused = gates[:, 0:1] * x_low + gates[:, 1:2] * x_high
        attention_high = torch.sigmoid(self.attention_head(fused))
        attention_low = 1.0 - attention_high
        return attention_high, attention_low


class AdaptiveHighLowFrequencyAggregation(nn.Module):
    """AAHF with a variable-resolution low-rank spatial projection.

    Classical Linformer learns E,F in R^(k x N), tying a layer to one image
    size. Adaptive average pooling is a deterministic rank-k projection that
    preserves the paper's O(N k) attention while supporting full-size KITTI
    images as well as 256x512 crops.
    """

    def __init__(self, disparity_channels=48, attention_dim=64, num_heads=4,
                 pooled_tokens=(8, 16)):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError('attention_dim must be divisible by num_heads')
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.pooled_tokens = tuple(pooled_tokens)

        input_dim = 2 * disparity_channels
        self.norm = nn.GroupNorm(1, input_dim)
        self.query = nn.Conv2d(input_dim, attention_dim, kernel_size=1, bias=False)
        self.key = nn.Conv2d(input_dim, attention_dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(input_dim, attention_dim, kernel_size=1, bias=False)
        self.output = nn.Conv2d(attention_dim, disparity_channels, kernel_size=1)
        self.residual = nn.Conv2d(input_dim, disparity_channels, kernel_size=1)

    def _as_heads(self, tensor):
        batch, _, height, width = tensor.shape
        return tensor.reshape(
            batch, self.num_heads, self.head_dim, height * width
        ).transpose(-1, -2)

    def forward(self, high_volume, low_volume):
        x = torch.cat([high_volume, low_volume], dim=1)
        normalized = self.norm(x)
        query = self._as_heads(self.query(normalized))

        key = F.adaptive_avg_pool2d(
            self.key(normalized), self.pooled_tokens)
        value = F.adaptive_avg_pool2d(
            self.value(normalized), self.pooled_tokens)
        key = self._as_heads(key)
        value = self._as_heads(value)

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores * (self.head_dim ** -0.5)
        attention = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = torch.matmul(attention, value)

        batch, _, height, width = high_volume.shape
        context = context.transpose(-1, -2).reshape(
            batch, self.attention_dim, height, width)
        return self.output(context) + self.residual(x)


class ConvolutionalHighLowFusion(nn.Module):
    """Paper ablation: AFFA followed by conventional 2D fusion."""

    def __init__(self, disparity_channels=48):
        super().__init__()
        self.fusion = nn.Sequential(
            BasicConv(2 * disparity_channels, disparity_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(disparity_channels, disparity_channels,
                      kernel_size=3, stride=1, padding=1),
        )

    def forward(self, high_volume, low_volume):
        return self.fusion(torch.cat([high_volume, low_volume], dim=1))


class MAFNet(nn.Module):
    """BANet-2D backbone/upsampler with paper-faithful AFFA and AAHF."""

    def __init__(self, args, use_affa=True, use_aahf=True):
        super().__init__()
        self.use_affa = bool(use_affa)
        self.fnet = FeatureNet(pretrained=True)
        self.cost_stem = BasicConv(48, 32, kernel_size=3, stride=1, padding=1)

        aggregation_kwargs = dict(
            in_channels=32, left_att=True, blocks=[4, 6, 8],
            expanse_ratio=4, backbone_channels=[64, 64, 192])
        self.cost_agg_high = Aggregation(**aggregation_kwargs)
        self.cost_agg_low = Aggregation(**aggregation_kwargs)
        self.affa = AdaptiveFrequencyDomainFilteringAttention()
        self.frequency_fusion = (
            AdaptiveHighLowFrequencyAggregation()
            if use_aahf else ConvolutionalHighLowFusion())

        self.stem_2 = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
            BasicConv(16, 16, kernel_size=3, stride=1, padding=1),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(16, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.spx = nn.ConvTranspose2d(
            2 * 16, 9, kernel_size=4, stride=2, padding=1)
        self.spx_2 = DeconvLayer(32, 16)
        self.spx_4 = nn.Sequential(
            BasicConv(64, 32, kernel_size=3, stride=1, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.build_cv = CostVolume()

    def forward(self, left, right, max_disp=192):
        features_left = self.fnet(left)
        features_right = self.fnet(right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), dim=1)
        features_right[0] = torch.cat((features_right[0], stem_4y), dim=1)

        correlation = self.build_cv(
            features_left[0], features_right[0], max_disp // 4)
        cost = self.cost_stem(correlation)

        if self.use_affa:
            attention_high, attention_low = self.affa(features_left)
        else:
            attention_high = cost.new_full(
                (cost.shape[0], 1, cost.shape[2], cost.shape[3]), 0.5)
            attention_low = 1.0 - attention_high

        high_volume = self.cost_agg_high(
            attention_high * cost, features_left)
        low_volume = self.cost_agg_low(
            attention_low * cost, features_left)
        cost = self.frequency_fusion(high_volume, low_volume)

        probability = F.softmax(cost, dim=1)
        disparity = disparity_regression(probability, max_disp // 4)

        upsample_features = self.spx_4(features_left[0])
        upsample_features = self.spx_2(upsample_features, stem_2x)
        upsample_weights = torch.softmax(self.spx(upsample_features), dim=1)
        disparity_up = context_upsample(disparity, upsample_weights)

        if self.training:
            disparity_linear = F.interpolate(
                disparity, left.shape[2:], mode='bilinear',
                align_corners=False)
            return [disparity_up * 4.0, disparity_linear * 4.0]
        return disparity_up * 4.0
