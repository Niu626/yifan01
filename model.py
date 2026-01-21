"""
    Transformer-UNet for illumination estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WindowAttention(nn.Module):
    """
    局部窗口注意力机制，用于保留多光源场景的局部差异
    """
    def __init__(self, d_model, nhead=8, window_size=7, dropout=0.1):
        super(WindowAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
    def window_partition(self, x, h, w):
        """
        将特征图分割成不重叠的窗口
        x: [H*W, B, C]
        """
        B = x.size(1)
        C = x.size(2)
        x = x.permute(1, 2, 0).reshape(B, C, h, w)  # [B, C, H, W]
        
        # 计算需要padding的大小
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = h, w
        
        # 分割成窗口
        num_windows_h = Hp // self.window_size
        num_windows_w = Wp // self.window_size
        
        x = x.view(B, C, num_windows_h, self.window_size, num_windows_w, self.window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, num_windows_h, num_windows_w, window_size, window_size, C]
        windows = windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, window_size^2, C]
        windows = windows.permute(1, 0, 2)  # [window_size^2, B*num_windows, C]
        
        return windows, (Hp, Wp), (num_windows_h, num_windows_w)
    
    def window_reverse(self, windows, Hp, Wp, h, w, num_windows):
        """
        将窗口特征还原回特征图
        """
        B = windows.size(1) // (num_windows[0] * num_windows[1])
        C = windows.size(2)
        
        windows = windows.permute(1, 0, 2)  # [B*num_windows, window_size^2, C]
        windows = windows.reshape(B, num_windows[0], num_windows[1], self.window_size, self.window_size, C)
        windows = windows.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, num_windows_h, window_size, num_windows_w, window_size]
        x = windows.view(B, C, Hp, Wp)
        
        # 裁剪padding
        if h != Hp or w != Wp:
            x = x[:, :, :h, :w]
        
        x = x.reshape(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        return x
    
    def forward(self, x, h, w):
        """
        x: [H*W, B, C]
        """
        # 窗口分割
        windows, (Hp, Wp), num_windows = self.window_partition(x, h, w)
        
        # 窗口内自注意力
        attn_out, _ = self.attn(windows, windows, windows)
        
        # 窗口还原
        x = self.window_reverse(attn_out, Hp, Wp, h, w, num_windows)
        return x


class LocalGlobalTransformerBlock(nn.Module):
    """
    局部-全局混合Transformer块
    结合窗口注意力和全局注意力，保留多光源场景的局部差异同时获取全局上下文
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, window_size=7, 
                 local_global_ratio=0.4):
        super(LocalGlobalTransformerBlock, self).__init__()
        # 全局注意力
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 局部窗口注意力
        self.local_attn = WindowAttention(d_model, nhead, window_size, dropout)
        # 可学习的融合权重（初始化为local_global_ratio）
        # 使用sigmoid确保alpha在[0, 1]范围内
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(local_global_ratio / (1 - local_global_ratio + 1e-8))))
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU替代ReLU，性能更好
        
        self.window_size = window_size

    def forward(self, src, h, w):
        """
        src: [H*W, B, C]
        h, w: 特征图的高度和宽度
        """
        # 计算alpha（通过sigmoid确保在[0, 1]范围内）
        alpha = torch.sigmoid(self.alpha_logit)
        
        # 1. 局部窗口注意力分支
        src_local = self.norm1(src)
        local_out = self.local_attn(src_local, h, w)
        src = src + self.dropout1(local_out) * alpha
        
        # 2. 全局注意力分支
        src_global = self.norm2(src)
        global_out, _ = self.global_attn(src_global, src_global, src_global)
        src = src + self.dropout2(global_out) * (1 - alpha)
        
        # 3. FFN
        src_ffn = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_ffn))))
        src = src + self.dropout3(src2)
        
        return src


class TransformerBlock(nn.Module):
    """
    保留原始TransformerBlock以兼容性，但建议使用LocalGlobalTransformerBlock
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class conv_block(nn.Module):  # 定义一个叫 conv_block 的小模块（子网络），输入通道数是 ch_in，输出通道数是 ch_out。   它是U-Net中反复出现的基本单元：卷积块
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()  # 调用父类（nn.Module）的初始化，这是PyTorch标准写法。
        self.conv = nn.Sequential(   # 使用 nn.Sequential 定义了一串顺序执行的网络操作。
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),              # 3×3卷积，输入通道ch_in，输出ch_out，保持尺寸（因为padding=1）。
            nn.ReLU(inplace=True),     # ReLU激活，增加非线性能力。
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),   # 再做一次 相同通道的卷积（ch_out -> ch_out），即两层卷积+激活，更深层次提取特征。
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # 定义前向传播。
        x = self.conv(x)  # 输入x经过上面顺序卷积操作，输出新的特征图。
        return x


class up_conv(nn.Module):  # 定义另一个模块叫 up_conv，是上采样模块，用来在解码器中恢复图像尺寸。
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Upsample是双线性插值，尺寸扩大2倍。
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)  # 上采样后再用3×3卷积调整特征图通道，并激活。
        )

    def forward(self, x): # 输入特征图x，执行上采样卷积流程。
        x = self.up(x)
        return x


# U-Net的任务是：给定受光照影响的输入图像（UVL格式），预测该图像在理想白平衡状态下应有的UV色度值。

# U-Net模型预测的UV值表示"假设无光照色偏时，图像应有的色度比"。

# 输入：受光照扭曲的UVL（由 rgb2uvl(input_rgb) 计算）。
# 输出：预测的UV，对应 假设无光照的色度比log（Rwb/Gwb）和log（Bwb/Gwb）。  Rwb：这个代表真实白平衡的R（由标签数据GT获得）.
class Transformer_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, d_model=512, nhead=8, num_layers=6, 
                 use_local_global_attention=True):
        super(Transformer_UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义2×2最大池化，下采样特征图。

        # 下面 conv1-8 是编码器， 压缩特征，提取深层信息
        # 4层下采样阶段，每一层都用 conv_block，通道数逐渐变大（特征图变深）。
        # 逐渐增加的通道数 代表提取了更多、更复杂的特征信息。  每个通道：以看作是一组特定类型的特征检测器，比如某个通道专门关注边缘，另一个通道专门关注纹理，还有通道关注局部对比度、颜色变化等等。
        # 通道数越多，意味着模型可以并行地捕捉更多种不同类型的视觉特征，从而使最终的光照推断更精准。
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        # Transformer部分 - 根据开关选择使用局部-全局混合注意力或原始全局注意力
        self.pos_encoder = PositionalEncoding(d_model)
        # 控制是否使用局部-全局混合注意力（用于消融实验）
        self.use_local_global = use_local_global_attention
        if self.use_local_global:
            self.transformer_blocks = nn.ModuleList([
                LocalGlobalTransformerBlock(d_model, nhead, window_size=7, 
                                          local_global_ratio=0.4) 
                for _ in range(num_layers)
            ])
        else:
            # 保留原始TransformerBlock作为备选
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, nhead) for _ in range(num_layers)
            ])
        self.proj_in = nn.Linear(512, d_model)
        self.proj_out = nn.Linear(d_model, 512)

        # 下面是解码器
        # 上采样恢复细节 + 跳跃连接
        self.Up4 = up_conv(ch_in=512, ch_out=256)  # 上采样8→7步
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)  # 上采样3→2步。
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)  # 上采样2→1步，基本恢复到初始大小。
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # 最后一层是1×1卷积，把通道数从64变成2（分别对应预测的 U 和 V ）。
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)   # U-Net的最后一层通过 Conv_1x1 输出一个2通道的特征图，对应 预测的UV值   注意： 这不是最终的光照估计图，而是对数色度比（log(R/G)和log(B/G)）


    def print_grad(self):   # 辅助调试函数，打印Conv1的信息，可以检查模型结构。
        print(self.Conv1)

    # 正式的前向传播（输入→输出）
    def forward(self, x):  # 定义输入x的前向计算流程。
        # 编码器阶段
        # 每次卷积后下采样，特征更深、尺寸更小，直到到达最底部。
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # Transformer处理
        b, c, h, w = x4.shape
        x4_flat = x4.view(b, c, -1).permute(2, 0, 1)  # [H*W, B, C]
        x4_flat = self.proj_in(x4_flat)
        x4_flat = self.pos_encoder(x4_flat)

        # 根据是否使用局部-全局混合注意力传递不同参数
        for transformer_block in self.transformer_blocks:
            if self.use_local_global:
                # LocalGlobalTransformerBlock需要h和w参数
                x4_flat = transformer_block(x4_flat, h, w)
            else:
                # 原始TransformerBlock不需要h和w
                x4_flat = transformer_block(x4_flat)

        x4_flat = self.proj_out(x4_flat)
        x4 = x4_flat.permute(1, 2, 0).reshape(b, c, h, w)

        # 解码器阶段（上采样+拼接+卷积）
        # 每次上采样后，将当前特征和编码器对应层（x7,x6,x5等）跳跃连接，然后再卷积融合。
        # 保证高分辨率信息不会丢失。
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 最后一层卷积输出2通道（U和V的预测），这是最终的输出结果。
        d1 = self.Conv_1x1(d2)  # 最终输出UV预测

        return d1
