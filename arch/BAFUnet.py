from abc import abstractmethod
import math
from CBMA import *
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from arch.swin_transformer import Mlp
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
size = 3
padding = 1
num = 32
band = 34
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, size, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, size, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class BAFUNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            lq_channels=31,
            rgb_channels=3
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.lq_channels = lq_channels
        self.rgb_channels = rgb_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.LeakyReLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.init_channels = model_channels
        ch = input_ch = int(channel_mult[0] * self.init_channels)

        self.input_blocks_denoise = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, size, padding=padding))]
        )

        # 2. LQ引导路径
        self.input_blocks_lq = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, lq_channels, ch, size, padding=padding))]
        )

        input_block_chans = [ch]
        ds = image_size  # 下采样比例

        # 构建下采样块
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                current_out_ch = int(mult * self.init_channels)

                # 三路使用相同的残差块结构
                denoise_layers = [
                    MSGABT(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=current_out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                lq_layers = [
                    MSGAB(
                        ch,
                        dropout,
                        out_channels=current_out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        attention_mode ="both"
                    )
                ]

                ch = current_out_ch

                # 添加注意力块


                self.input_blocks_denoise.append(TimestepEmbedSequential(*denoise_layers))
                self.input_blocks_lq.append(nn.Sequential(*lq_layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                # 三路下采样
                denoise_down = TimestepEmbedSequential(
                    MSGABT(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        attention_mode="both"
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )

                lq_down = nn.Sequential(
                    MSGAB(
                        ch,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        attention_mode="channel"
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )


                self.input_blocks_denoise.append(denoise_down)
                self.input_blocks_lq.append(lq_down)

                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        # 中间块（三路融合后处理）
        self.middle_block = nn.Sequential(
            MSGAB(
                ch * 2,  # 融合三路特征，通道数三倍
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                attention_mode="both"
            ),
        )
        ch = ch * 2

        # 上采样部分
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    MSGAB(
                        ch + ich * 2,  # 输入通道数（当前特征 + 三路跳连特征）
                        dropout,
                        out_channels=int(self.init_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        attention_mode = 'both'
                    )
                ]
                ch = int(self.init_channels * mult)

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(
                            MSGAB(
                                ch,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_scale_shift_norm=use_scale_shift_norm,
                                attention_mode='both'
                            )
                        )
                    else:
                        layers.append(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                    ds *= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.LeakyReLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, size, padding=padding)),
        )

    def forward(self, x, rgb, lq, timesteps):

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        # 三路处理
        hs_denoise = []
        h_denoise = x.type(self.dtype)
        for module in self.input_blocks_denoise:
            h_denoise = module(h_denoise, emb)
            hs_denoise.append(h_denoise)

        hs_lq = []
        #h_lq = lq.type(self.dtype)
        h_lq = th.cat([lq,rgb], dim=1)
        h_lqrgb = h_lq.type(self.dtype)
        for module in self.input_blocks_lq:
            h_lqrgb = module(h_lqrgb)
            hs_lq.append(h_lqrgb)
        h = th.cat([h_denoise, h_lqrgb], dim=1)#, h_rgb
        h = self.middle_block(h)
        upsample_outputs = []
        for i, module in enumerate(self.output_blocks):
            denoise_skip = hs_denoise.pop()
            lq_skip = hs_lq.pop()
            skip = th.cat([denoise_skip, lq_skip], dim=1)
            h = th.cat([h, skip], dim=1)
            h = module(h)
            upsample_outputs.append(h)

        return h, upsample_outputs

    def convert_to_fp16(self):
        self.input_blocks_denoise.apply(convert_module_to_f16)
        self.input_blocks_lq.apply(convert_module_to_f16)

        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks_denoise.apply(convert_module_to_f32)
        self.input_blocks_lq.apply(convert_module_to_f32)

        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


# 新增：不含时间嵌入的残差块
class MSGAB(nn.Module):
    """
    不接收时间嵌入的残差块，用于引导路径
    通过 attention_mode 参数控制注意力机制类型
    """

    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            kernel_size=7,
            reduction_ratio=17,
            up=False,
            down=False,
            attention_mode="none",
            mlp_ratio=4.0,
            groups_list=[1, band],
            groups_list1 = [1, band]
            # 新增参数：控制注意力机制类型
    ):
        super().__init__()
        self.channels = channels
        self.groups_list = groups_list
        self.groups_list1 = groups_list1
        self.group_convs = nn.ModuleList()
        self.group_convs1 = nn.ModuleList()
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.attention_mode = attention_mode  # 存储注意力模式
        #self.norm = nn.LayerNorm(normalized_shape)
        # 根据注意力模式决定是否创建注意力模块
        if attention_mode in ["channel", "both"]:
            self.catt = ChannelAttention(self.out_channels, reduction_ratio)
        if attention_mode in ["spatial", "both"]:
            self.sptt = SpatialAttention(kernel_size)

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.LeakyReLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=kernel_size // 2),
        )

        self.updown = up or down
        self.LayerNorm2d = LayerNorm2d(self.channels)
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=kernel_size // 2)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=kernel_size // 2
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        mlp_hidden_dim = int(self.out_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.out_channels*2,  # 使用self.out_channels而不是channels
            hidden_features=mlp_hidden_dim,
            out_features=self.out_channels,  # 添加输出特征数
            act_layer=nn.LeakyReLU,
            drop=dropout
        )
        for g in groups_list:
            self.group_convs.append(nn.Sequential(
                nn.Conv2d(channels, self.out_channels, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list), len(groups_list), 1),
            nn.Softmax(dim=1)
        )
        for g in groups_list1:
            self.group_convs1.append(nn.Sequential(
                nn.Conv2d(self.out_channels , self.out_channels*2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels*2),
                nn.Conv2d(self.out_channels*2, self.out_channels*2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list), len(groups_list), 1),
            nn.Softmax(dim=1)
        )
        self.attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list)*2, len(groups_list)*2, 1),
            nn.Softmax(dim=1)
        )
        # 可变形卷积
        self.offset_conv = nn.Conv2d(self.out_channels, 2 * 3 * 3, kernel_size=3, padding=1)
        # 输出层
        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2)
        )
        self.back = nn.Sequential(
            nn.Conv2d(self.out_channels*3, self.out_channels, kernel_size=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        self.back1 = nn.Sequential(
            nn.Conv2d(self.out_channels*2, self.out_channels, kernel_size=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        self.sc = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=1),
        )
    def forward(self, x):

            group_bothoutputs = []
            h1 = self.scale1(x)
            h = self.sptt(h1) * h1
            for conv in self.group_convs1:
                group_bothoutputs.append(conv(h))
            # 注意力加权融合
            attn_weights = self.attention1(torch.cat([
                F.adaptive_avg_pool2d(out, 1) for out in group_bothoutputs
            ], dim=1))
            h = sum(attn_weights[:, i:i + 1] * group_bothoutputs[i]
                               for i in range(len(self.groups_list1)))
            h = self.mlp(h)
            return  h + self.skip_connection(x)

class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x):
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
            return x
class LearnableUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = LayerNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        return x

class MSGABT(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=7,
        reduction_ratio=17,
        attention_mode="none",
        mlp_ratio=4.0,
        groups_list=[1, band],
        groups_list1=[1, band]
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.attention_mode = attention_mode
        self.groups_list = groups_list
        self.groups_list1 = groups_list1
        self.group_convs = nn.ModuleList()
        self.group_convs1 = nn.ModuleList()
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, size, padding=padding),
        )
        self.LayerNorm2d = LayerNorm2d(self.channels)
        self.updown = up or down

        self.catt = ChannelAttention(self.out_channels, reduction_ratio)

        self.sptt = SpatialAttention(kernel_size)
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        mlp_hidden_dim = int(self.out_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.out_channels*2,  # 使用self.out_channels而不是channels
            hidden_features=mlp_hidden_dim,
            out_features=self.out_channels,  # 添加输出特征数
            act_layer=nn.LeakyReLU,
            drop=dropout
        )
        for g in groups_list:
            self.group_convs.append(nn.Sequential(
                nn.Conv2d(channels, self.out_channels, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list), len(groups_list), 1),
            nn.Softmax(dim=1)
        )
        for g in groups_list1:
            self.group_convs1.append(nn.Sequential(
                nn.Conv2d(self.out_channels , self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels * 2),
                nn.Conv2d(self.out_channels * 2, self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list), len(groups_list), 1),
            nn.Softmax(dim=1)
        )
        self.attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list) * 2, len(groups_list) * 2, 1),
            nn.Softmax(dim=1)
        )
        self.emb_layers = nn.Sequential(
            nn.LeakyReLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, size, padding=padding)
            ),
        )
        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        self.back = nn.Sequential(
            nn.Conv2d(self.out_channels * 3, self.out_channels, kernel_size=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        self.back1 = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):

        #h = self.in_layers(x)

            group_bothoutputs = []
            h1 = self.scale1(x)
            h1 = self.sptt(h1) * h1
            #h2 = self.scale2(h1)
            #h21 = nn.functional.interpolate(h2, scale_factor=2, mode='bicubic', align_corners=False)
            #h21 = self.sptt(h21) * h21
            emb_out = self.emb_layers(emb).type(h1.dtype)
            while len(emb_out.shape) < len(h1.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                print('1111')
            else:
                h1 = h1 + emb_out
                #h21 = h21 + emb_out

            #h = torch.cat([h1, h21], dim=1)

            h= h1
            for conv in self.group_convs1:
                group_bothoutputs.append(conv(h))
            # 注意力加权融合
            attn_weights = self.attention1(torch.cat([
                F.adaptive_avg_pool2d(out, 1) for out in group_bothoutputs
            ], dim=1))
            h11 = sum(attn_weights[:, i:i + 1] * group_bothoutputs[i]
                               for i in range(len(self.groups_list1)))

            #x1 = self.back1(h11)
            h = self.mlp(h11)

            return h + self.skip_connection(x)