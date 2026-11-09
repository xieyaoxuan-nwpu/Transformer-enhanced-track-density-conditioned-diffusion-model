'''
V4模型 - Transformer编码器版本
核心创新: 将UNet编码器部分的ResBlock+Self-Attention完全替换为Transformer Block

创新点:
1. DualPathESPCN: PixelShuffle + 反卷积双路径融合 (保留)
2. TransformerEncoderBlock: 替代原有的ResBlock，使用Multi-Head Self-Attention + FFN
   - 编码器使用纯Transformer块
   - 解码器保留卷积结构以保证空间精度
'''

import math
import torch
from torch import nn
from torch.nn import functional as F


# ==========================================
# 基础组件
# ==========================================
class Swish(nn.Module):
    """激活函数 - SiLU/Swish"""
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU()


def get_num_groups(channels, max_groups=32):
    """动态计算GroupNorm的num_groups，确保能整除channels"""
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ==========================================
# 时间编码模块
# ==========================================
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


# ==========================================
# 创新点1: DualPathESPCN (保留)
# ==========================================
class DualPathESPCN(nn.Module):
    """
    双路径超分辨率模块:
    - 路径1: PixelShuffle (保留高频细节)
    - 路径2: 反卷积/转置卷积 (平滑结构)
    - 融合: 可学习权重融合两条路径
    """
    def __init__(self, in_ch=1, upscale_factor=2):
        super(DualPathESPCN, self).__init__()
        mid_ch = 64
        out_ch = 1
        
        # 共享特征提取层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=5, stride=1, padding=2),
            get_activation(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            get_activation(),
        )
        
        # 路径1: PixelShuffle (亚像素卷积)
        self.pixel_shuffle_path = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            get_activation(),
            nn.Conv2d(mid_ch, out_ch * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
        )
        
        # 路径2: 反卷积 (转置卷积)
        self.deconv_path = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            get_activation(),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=upscale_factor, padding=1),
        )
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # 融合后的精细化卷积
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch * 2, kernel_size=3, stride=1, padding=1),
            get_activation(),
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        feat = self.shared_conv(x)
        path1 = self.pixel_shuffle_path(feat)
        path2 = self.deconv_path(feat)
        weights = F.softmax(self.fusion_weight, dim=0)
        combined = torch.cat([path1 * weights[0], path2 * weights[1]], dim=1)
        out = self.fusion_refine(combined)
        return torch.sigmoid(out)


# ==========================================
# 创新点2: Transformer编码器块 (核心改进)
# ==========================================
class PositionalEncoding2D(nn.Module):
    """2D位置编码 - 用于图像特征"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        device = x.device
        
        # 生成位置编码
        y_pos = torch.arange(H, device=device).float()
        x_pos = torch.arange(W, device=device).float()
        
        # 计算编码维度
        dim_t = C // 4
        if dim_t == 0:
            return x
            
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_t, device=device).float() / dim_t))
        
        # 计算sin/cos编码
        pos_y = y_pos[:, None] * inv_freq[None, :]  # [H, dim_t]
        pos_x = x_pos[:, None] * inv_freq[None, :]  # [W, dim_t]
        
        pe_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [H, dim_t*2]
        pe_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)  # [W, dim_t*2]
        
        # 扩展到完整尺寸
        pe_y = pe_y[:, None, :].expand(H, W, -1)  # [H, W, dim_t*2]
        pe_x = pe_x[None, :, :].expand(H, W, -1)  # [H, W, dim_t*2]
        
        pe = torch.cat([pe_y, pe_x], dim=-1)  # [H, W, dim_t*4]
        
        # 调整到正确的通道数
        if pe.shape[-1] < C:
            pe = F.pad(pe, (0, C - pe.shape[-1]))
        elif pe.shape[-1] > C:
            pe = pe[..., :C]
            
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return x + pe


class EfficientAttention(nn.Module):
    """
    高效注意力机制 - 使用线性注意力降低计算复杂度
    从O(N^2)降低到O(N)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """x: [B, N, C] where N = H*W"""
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        
        # 线性注意力: 使用ELU+1作为核函数
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 计算 k^T @ v 先，然后与 q 相乘
        # [B, heads, head_dim, N] @ [B, heads, N, head_dim] -> [B, heads, head_dim, head_dim]
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # [B, heads, N, head_dim] @ [B, heads, head_dim, head_dim] -> [B, heads, N, head_dim]
        qkv_out = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # 归一化
        k_sum = k.sum(dim=2, keepdim=True)  # [B, heads, 1, head_dim]
        qk_sum = (q * k_sum).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]
        
        out = qkv_out / (qk_sum + 1e-6)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class FeedForward(nn.Module):
    """前馈网络 - Transformer的FFN部分"""
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块 - 替代原有的ResBlock
    结构: LayerNorm -> MHSA -> 残差 -> LayerNorm -> FFN -> 残差
    
    增加时间嵌入的注入
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0., tdim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, 
            attn_drop=dropout, 
            proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        
        # 时间嵌入投影
        if tdim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(tdim, dim)
            )
        else:
            self.time_mlp = None
            
        # 可学习的缩放因子
        self.gamma1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, x, temb=None):
        """
        x: [B, C, H, W]
        temb: [B, tdim] 时间嵌入
        """
        B, C, H, W = x.shape
        
        # 转换为序列格式 [B, N, C]
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Self-Attention
        x_seq = x_seq + self.gamma1 * self.attn(self.norm1(x_seq))
        
        # 注入时间嵌入
        if self.time_mlp is not None and temb is not None:
            time_emb = self.time_mlp(temb)  # [B, C]
            x_seq = x_seq + time_emb.unsqueeze(1)
        
        # FFN
        x_seq = x_seq + self.gamma2 * self.ffn(self.norm2(x_seq))
        
        # 转换回图像格式 [B, C, H, W]
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class TransformerDownBlock(nn.Module):
    """
    Transformer下采样块 - 包含多个Transformer块 + 下采样
    返回下采样前的特征用于跳跃连接
    """
    def __init__(self, in_ch, out_ch, tdim, num_blocks=2, num_heads=8, dropout=0., downsample=True):
        super().__init__()
        
        # 通道调整
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # 位置编码
        self.pos_enc = PositionalEncoding2D(out_ch)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=out_ch, 
                num_heads=num_heads, 
                mlp_ratio=4., 
                dropout=dropout,
                tdim=tdim
            ) for _ in range(num_blocks)
        ])
        
        # 下采样 (使用卷积)
        self.do_downsample = downsample
        if downsample:
            self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, temb):
        x = self.proj_in(x)
        x = self.pos_enc(x)
        
        for block in self.blocks:
            x = block(x, temb)
        
        if self.do_downsample:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class TransformerMiddleBlock(nn.Module):
    """
    Transformer中间块 - UNet的bottleneck
    """
    def __init__(self, ch, tdim, num_blocks=2, num_heads=8, dropout=0.):
        super().__init__()
        
        self.pos_enc = PositionalEncoding2D(ch)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=ch, 
                num_heads=num_heads, 
                mlp_ratio=4., 
                dropout=dropout,
                tdim=tdim
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x, temb):
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, temb)
        return x


# ==========================================
# 解码器部分 (保留卷积结构)
# ==========================================
class ConvResBlock(nn.Module):
    """卷积残差块 - 用于解码器"""
    def __init__(self, in_ch, out_ch, tdim, dropout=0.):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class UpSample(nn.Module):
    """上采样模块"""
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        
    def forward(self, x, temb=None):
        x = self.up(x)
        x = self.conv(x)
        return x


class ConvUpBlock(nn.Module):
    """卷积上采样块 - 用于解码器"""
    def __init__(self, in_ch, out_ch, tdim, num_blocks=2, dropout=0., upsample=True):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ConvResBlock(
                    in_ch if i == 0 else out_ch, 
                    out_ch, 
                    tdim, 
                    dropout
                )
            )
        
        self.do_upsample = upsample
        if upsample:
            self.upsample = UpSample(out_ch)
        
    def forward(self, x, skip, temb):
        # 确保尺寸匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, temb)
        if self.do_upsample:
            x = self.upsample(x, temb)
        return x


# ==========================================
# TDI编码器 (使用Transformer)
# ==========================================
class TransformerEncoderTDI(nn.Module):
    """
    使用Transformer的TDI编码器
    """
    def __init__(self, ch, in_ch=1, out_ch=1, num_blocks=2, num_heads=4):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.GroupNorm(get_num_groups(ch), ch),
            nn.SiLU(),
        )
        
        self.pos_enc = PositionalEncoding2D(ch)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=ch, 
                num_heads=num_heads, 
                mlp_ratio=2., 
                dropout=0.,
                tdim=None
            ) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Sequential(
            nn.GroupNorm(get_num_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.output_proj(x)
        
        return x + identity


# ==========================================
# V4 UNet - Transformer编码器版本
# ==========================================
class UNet(nn.Module):
    """
    V4 UNet - Transformer编码器版本
    
    核心改进:
    - 编码器: 使用TransformerEncoderBlock替代ResBlock+AttnBlock
    - 解码器: 保留卷积结构以保证空间精度
    - 中间层: 使用Transformer块
    """
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_levels = len(ch_mult)
        
        # 时间编码
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        
        # 创新点1: 双路径ESPCN
        self.dual_path_espcn = DualPathESPCN(in_ch=1, upscale_factor=2)
        
        # TDI编码器 (使用Transformer)
        self.fibre_density_encoder = TransformerEncoderTDI(
            ch=ch, in_ch=1, out_ch=1, num_blocks=2, num_heads=4
        )
        
        # 输入头
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        
        # ============ 编码器 (Transformer) ============
        self.encoder_blocks = nn.ModuleList()
        
        now_ch = ch
        self.encoder_channels = []  # 记录每层的通道数用于跳跃连接
        
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            is_last = (i == len(ch_mult) - 1)
            
            # Transformer编码块
            self.encoder_blocks.append(
                TransformerDownBlock(
                    in_ch=now_ch,
                    out_ch=out_ch,
                    tdim=tdim,
                    num_blocks=num_res_blocks,
                    num_heads=max(4, out_ch // 16),
                    dropout=dropout,
                    downsample=not is_last  # 最后一层不下采样
                )
            )
            
            self.encoder_channels.append(out_ch)
            now_ch = out_ch
        
        # ============ 中间层 (Transformer) ============
        self.middle_block = TransformerMiddleBlock(
            ch=now_ch,
            tdim=tdim,
            num_blocks=2,
            num_heads=max(4, now_ch // 16),
            dropout=dropout
        )
        
        # ============ 解码器 (卷积) ============
        self.decoder_blocks = nn.ModuleList()
        
        # 反转通道列表用于解码器
        decoder_in_channels = list(reversed(self.encoder_channels))
        
        for i in range(len(ch_mult)):
            skip_ch = decoder_in_channels[i]
            is_last = (i == len(ch_mult) - 1)
            
            if is_last:
                out_ch = ch
            else:
                out_ch = decoder_in_channels[i + 1] if i + 1 < len(decoder_in_channels) else ch
            
            self.decoder_blocks.append(
                ConvUpBlock(
                    in_ch=now_ch + skip_ch,
                    out_ch=out_ch,
                    tdim=tdim,
                    num_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample=not is_last  # 最后一层不上采样
                )
            )
            now_ch = out_ch
        
        # 输出尾
        self.tail = nn.Sequential(
            nn.GroupNorm(get_num_groups(now_ch), now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1),
        )

    def forward(self, x, t, labels, fibre_density):
        # 时间编码
        temb = self.time_embedding(t)
        
        # TDI编码
        fibre_density = self.fibre_density_encoder(fibre_density)
        
        # 双路径ESPCN处理低分辨率条件
        labels = self.dual_path_espcn(labels)
        
        # 拼接输入
        x = torch.cat((x, labels, fibre_density), dim=1)
        
        # 输入头
        h = self.head(x)
        
        # ============ 编码器 ============
        encoder_features = []
        for encoder_block in self.encoder_blocks:
            h_skip, h = encoder_block(h, temb)
            encoder_features.append(h_skip)
        
        # ============ 中间层 ============
        h = self.middle_block(h, temb)
        
        # ============ 解码器 ============
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = encoder_features.pop()
            h = decoder_block(h, skip, temb)
        
        # 输出
        h = self.tail(h)
        
        return h
