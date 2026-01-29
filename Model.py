import os
import json
import math
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

def make_gn(C: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, C)
    while C % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=C)

class Conv3x3Reflect(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=bias)
    def forward(self, x):
        return self.conv(self.pad(x))

class FakeMaskGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv3x3Reflect(in_channels, 32),
            Norm2d(32),
            nn.ReLU(inplace=True),
            Conv3x3Reflect(32, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.conv_block(img)
    
    
class ResConvBlock(nn.Module):
    """Residual convolution block (matching Keras Attention ResUNet)"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # Residual connection
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)

        out = self.dropout(out)
        return out

def Norm2d(C: int, max_groups: int = 32, eps: float = 1e-5):
    g = min(max_groups, C)
    while g > 1 and (C % g != 0):
        g -= 1
    return nn.GroupNorm(g, C, eps=eps)

class TwoSelfAttnFuse(nn.Module):
    
    def __init__(self, dim, heads=2, window_size=(4, 4), stride=(2, 2),
                 fusion="soft_add", eps=1e-6, tau_init: float = 1.0):
        super().__init__()
        assert dim % heads == 0
        if isinstance(window_size, int): window_size = (window_size, window_size)
        if isinstance(stride, int):      stride = (stride, stride)
        self.h = heads
        self.d = dim // heads
        self.wh, self.ww = window_size
        self.sh, self.sw = stride
        self.scale = self.d ** -0.5
        self.fusion = fusion
        self.eps = eps

        self.qkv_x = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.qk_z  = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.proj  = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self._pa     = nn.Parameter(torch.tensor(0.0))
        self._palpha = nn.Parameter(torch.tensor(0.0))
        self._pbeta  = nn.Parameter(torch.tensor(0.0))
        self._ptau   = nn.Parameter(torch.tensor(tau_init))

        self.pad_h = max(0, self.wh - self.sh)
        self.pad_w = max(0, self.ww - self.sw)
        self.pad = (self.pad_w // 2, self.pad_w - self.pad_w // 2,
                    self.pad_h // 2, self.pad_h - self.pad_h // 2)

    def _unfold(self, x, kernel_size, stride):
        x_pad = F.pad(x, self.pad, mode="reflect")
        cols = F.unfold(x_pad, kernel_size=kernel_size, stride=stride)
        Hp, Wp = x_pad.shape[-2:]
        Nh = (Hp - kernel_size[0]) // stride[0] + 1
        Nw = (Wp - kernel_size[1]) // stride[1] + 1
        return cols, (Hp, Wp, Nh, Nw)

    def _fold(self, cols, out_hw, kernel_size, stride):
        Hp, Wp = out_hw
        return F.fold(cols, output_size=(Hp, Wp), kernel_size=kernel_size, stride=stride)

    def _cols_to_heads(self, cols, B, C, Nwin, wh, ww):
        Ntok = wh * ww
        t = cols.transpose(1, 2).contiguous().view(B * Nwin, C, wh, ww)
        t = t.view(B * Nwin, self.h, self.d, wh, ww).permute(0, 1, 3, 4, 2)
        return t.reshape(B * Nwin, self.h, Ntok, self.d)

    def forward(self, x_feat: torch.Tensor, z_feat: torch.Tensor):
        B, C, H, W = x_feat.shape
        wh, ww, sh, sw = self.wh, self.ww, self.sh, self.sw
        eps = self.eps

        # Stream X (Img) -> QKV
        qx, kx, vx = torch.chunk(self.qkv_x(x_feat), 3, dim=1)
        qx_cols, (Hp, Wp, Nh, Nw) = self._unfold(qx, (wh, ww), (sh, sw))
        kx_cols, _ = self._unfold(kx, (wh, ww), (sh, sw))
        vx_cols, _ = self._unfold(vx, (wh, ww), (sh, sw))

        Nwin = Nh * Nw                   
        Ntok = wh * ww                      

        qxh = self._cols_to_heads(qx_cols, B, C, Nwin, wh, ww)   # [B*Nwin,h,Ntok,d]
        kxh = self._cols_to_heads(kx_cols, B, C, Nwin, wh, ww)
        vxh = self._cols_to_heads(vx_cols, B, C, Nwin, wh, ww)
        
        # Stream Z (Fake/GT) -> QK
        qz, kz = torch.chunk(self.qk_z(z_feat), 2, dim=1)
        qz_cols, _ = self._unfold(qz, (wh, ww), (sh, sw))
        kz_cols, _ = self._unfold(kz, (wh, ww), (sh, sw))
        qzh = self._cols_to_heads(qz_cols, B, C, Nwin, wh, ww)
        kzh = self._cols_to_heads(kz_cols, B, C, Nwin, wh, ww)
        
        # 1st stage
        logits_x = (qxh @ kxh.transpose(-2, -1)) * self.scale
        logits_x = logits_x - logits_x.amax(dim=-1, keepdim=True)
        logits_z = (qzh @ kzh.transpose(-2, -1)) * self.scale
        logits_z = logits_z - logits_z.amax(dim=-1, keepdim=True)
        
        beta = F.softplus(self._pbeta) + 1e-6
        A_local = (logits_x + beta * logits_z).softmax(dim=-1)   # [B*Nwin,h,Ntok,Ntok]
        
        out_win = A_local @ vxh                                  # [B*Nwin,h,Ntok,d]

        #2nd stage
        out_win_b = out_win.view(B, Nwin, self.h, Ntok, self.d)      # [B,Nwin,h,Ntok,d]
        out_win_b = out_win_b.permute(0, 2, 1, 3, 4).contiguous()     # [B,h,Nwin,Ntok,d]
        L = Nwin * Ntok
        qg = out_win_b.view(B, self.h, L, self.d)                    # [B,h,L,d]
        kg = qg                                                       # self-attn
        vg = qg

        logits_g = (qg @ kg.transpose(-2, -1)) * self.scale          # [B,h,L,L]
        logits_g = logits_g - logits_g.amax(dim=-1, keepdim=True)
        A_global = logits_g.softmax(dim=-1)                          # [B,h,L,L]

        out_global = A_global @ vg                                   # [B,h,L,d]

        out_global = out_global.view(B, self.h, Nwin, Ntok, self.d)  # [B,h,Nwin,Ntok,d]
        out_global = out_global.permute(0, 2, 1, 3, 4).contiguous()  # [B,Nwin,h,Ntok,d]
        out_global = out_global.view(B * Nwin, self.h, Ntok, self.d) # [B*Nwin,h,Ntok,d]

        out_win = out_global

        out_win = out_win.view(B * Nwin, self.h, self.wh, self.ww, self.d)\
                         .permute(0, 1, 4, 2, 3).contiguous()
        out_win = out_win.view(B * Nwin, C, self.wh, self.ww)
        out_cols = out_win.view(B, Nwin, C, self.wh, self.ww)\
                           .permute(0, 2, 3, 4, 1).contiguous()
        out_cols = out_cols.view(B, C * self.wh * self.ww, Nwin)
        out_sum = self._fold(out_cols, (Hp, Wp), (wh, ww), (sh, sw))
        
        ones = x_feat.new_ones((B, 1, H, W))
        ones_cols, _ = self._unfold(ones, (wh, ww), (sh, sw))
        ones_sum = self._fold(ones_cols, (Hp, Wp), (wh, ww), (sh, sw))
        out_avg = out_sum / (ones_sum + eps)
        Lp, Rp, Tp, Bp = self.pad
        out_crop = out_avg[..., Tp:Hp - Bp, Lp:Wp - Rp]
        a = 2.0 * torch.sigmoid(self._pa)

        fused = x_feat + a * self.proj(out_crop)
        return fused       
    
class ConsensusAttnResUNetStudent(nn.Module):
    """
    Optimized Siamese Consensus ResUNet.
    Implementation Details:
    1. Parallelized Encoder: Image and Mask are batched together for 1-pass encoding.
    2. Memory Efficient: Detach logic integrated into the batched flow.
    3. Vectorized Gating: Uses a parameter vector for alpha weights.
    """
    def __init__(self, in_channels=1, n_filters=32, dropout=0.1,
                 attn_window=(8, 8), attn_stride=(1, 1),
                 attn_heads=2,
                 stop_grad_fake_encoder=False):
        super().__init__()
        self.stop_grad_fake_encoder = stop_grad_fake_encoder
        self.fake_mask_gen = FakeMaskGenerator(in_channels=in_channels, out_channels=1)

        # Shared Encoder Blocks (Siamese)
        self.enc1 = ResConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        
        bott_ch = n_filters * 16
        self.bottleneck = ResConvBlock(n_filters * 8, bott_ch, dropout)
        
        # Optimized Bottleneck Attention
        self.bott_attn = TwoSelfAttnFuse(
            dim=bott_ch, heads=attn_heads,
            window_size=attn_window, stride=attn_stride,
        )

        # Decoder Blocks
        self.up4 = nn.ConvTranspose2d(bott_ch, n_filters * 8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)

        self.dec4 = ResConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.dec3 = ResConvBlock(n_filters * 8,  n_filters * 4, dropout)
        self.dec2 = ResConvBlock(n_filters * 4,  n_filters * 2, dropout)
        self.dec1 = ResConvBlock(n_filters * 2,  n_filters,     dropout)
        
        self.out = nn.Conv2d(n_filters, 1, 1)
        
        # Learnable Consensus Parameters
        self.alphas = nn.Parameter(torch.ones(4) * 0.5)

    def _encode_stream(self, x_combined):
        """Processes a batched tensor through the encoder levels."""
        c1 = self.enc1(x_combined)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        c4 = self.enc4(self.pool3(c3))
        cb = self.bottleneck(self.pool4(c4))
        return c1, c2, c3, c4, cb

    def forward(self, x):
        fake_mask = self.fake_mask_gen(x)

        if self.stop_grad_fake_encoder:
            # Separate flows to maintain gradient isolation for weights
            e1, e2, e3, e4, b = self._encode_stream(x)
            with torch.no_grad():
                z1, z2, z3, z4, fb = self._encode_stream(fake_mask)
        else:
            # Parallelized Batch Flow: Process both Image and Mask in one kernel pass
            combined = torch.cat([x, fake_mask], dim=0) # [2B, C, H, W]
            c1, c2, c3, c4, cb = self._encode_stream(combined)
            
            # Split streams [Image | Mask]
            e1, z1 = torch.chunk(c1, 2, dim=0)
            e2, z2 = torch.chunk(c2, 2, dim=0)
            e3, z3 = torch.chunk(c3, 2, dim=0)
            e4, z4 = torch.chunk(c4, 2, dim=0)
            b, fb = torch.chunk(cb, 2, dim=0)

        # Consensus Attention Bottleneck
        b_fused = self.bott_attn(b, fb)

        # Decoder with Vectorized Consensus Gating
        def consensus_up(decoder_in, skip_feat, mask_feat, up_layer, dec_layer, alpha):
            up = up_layer(decoder_in)
            gated_skip = skip_feat * (1.0 - alpha * mask_feat)
            return dec_layer(torch.cat([up, gated_skip], dim=1))

        d4 = consensus_up(b_fused, e4, z4, self.up4, self.dec4, self.alphas[0])
        d3 = consensus_up(d4, e3, z3, self.up3, self.dec3, self.alphas[1])
        d2 = consensus_up(d3, e2, z2, self.up2, self.dec2, self.alphas[2])
        d1 = consensus_up(d2, e1, z1, self.up1, self.dec1, self.alphas[3])

        logits = self.out(d1)
        return logits
    



