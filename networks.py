import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward
from torchinfo import summary
from einops.layers.torch import Rearrange
from torchprofile import profile_macs
from ptflops import get_model_complexity_info





class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x



class MultiFrequency_Unbiased(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(MultiFrequency_Unbiased, self).__init__()
        self.groups = groups

        self.input_shape = 32  # change!!!!!it!!!!!!manually!!!!!!
        self.in_channels = in_channels

        self.locMap = nn.Parameter(torch.rand(self.input_shape, self.input_shape // 2 + 1))

        self.lambda_base = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.conv_layer_down55 = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                 out_channels=out_channels * 2,
                                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups,
                                                 bias=False, padding_mode='reflect')
        self.conv_layer_down55_shift = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                       out_channels=out_channels * 2,
                                                       kernel_size=3, stride=1, padding=2, dilation=2,
                                                       groups=self.groups, bias=False, padding_mode='reflect')

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.img_freq = None
        self.distill = None

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        locMap = self.locMap.expand_as(ffted[:, :1, :, :])  # B 1 H' W'
        ffted_copy = ffted.clone()

        cat_img_mask_freq = torch.cat((ffted[:, :self.in_channels, :, :],
                                       ffted[:, self.in_channels:, :, :],
                                       locMap), dim=1)

        ffted = self.conv_layer_down55(cat_img_mask_freq)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        ffted = self.relu(ffted)

        locMap_shift = torch.fft.fftshift(locMap, dim=-2)  ## ONLY IF NOT SHIFT BACK

        # REPEAT CONV
        cat_img_mask_freq1 = torch.cat((ffted[:, :self.in_channels, :, :],
                                        ffted[:, self.in_channels:, :, :],
                                        locMap_shift), dim=1)

        ffted = self.conv_layer_down55_shift(cat_img_mask_freq1)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        lambda_base = torch.sigmoid(self.lambda_base)

        ffted = ffted_copy * lambda_base + ffted * (1 - lambda_base)

        # irfft
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        epsilon = 0.5
        output = output - torch.mean(output) + torch.mean(x)
        output = torch.clip(output, float(x.min() - epsilon), float(x.max() + epsilon))

        self.distill = output  # for self perc
        return output



class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # B, N, C
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous().view(B, -1, self.num_heads * self.head_dim)

        out = self.fc(out).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

class DAFA(nn.Module):
    def __init__(self, dim=256, height=2, reduction=8, num_heads=4):
        super(DAFA, self).__init__()
        self.height = height
        self.attention = MultiHeadSelfAttention(dim, num_heads)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn_out = self.attention(feats_sum)
        return attn_out



class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #hidden_features = int(2 * hidden_features / 3)
        hidden_features = in_features // 2
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCoordGate(nn.Module):
    def __init__(self, enc_channels, out_channels, size: list = [64, 64], enctype='pos', encoding_layers=3):
        super(UnifiedCoordGate, self).__init__()
        self.enctype = enctype
        self.enc_channels = enc_channels
        self.out_channels = out_channels

        if enctype == 'pos':
            self.position_encoder = PositionEncoder(size, encoding_layers=encoding_layers)
            self.spatial_gate_conv = nn.Conv2d(64, enc_channels, 1)
            self.channel_gate_conv = nn.Conv2d(enc_channels, enc_channels, 1)
            self.final_conv = nn.Conv2d(enc_channels, out_channels, 1, padding='same')
            self.relu = nn.ReLU()

        elif enctype == 'map' or enctype == 'bilinear':
            self.spatial_gate_conv = nn.Conv2d(enc_channels, enc_channels, 1)
            self.channel_gate_conv = nn.Conv2d(enc_channels, enc_channels, 1)
            self.final_conv = nn.Conv2d(enc_channels, out_channels, 1, padding='same')
            self.relu = nn.ReLU()
            self.sample = kwargs.get('downsample', [1, 1])

    def forward(self, x):
        if self.enctype == 'pos':
            gate = self.position_encoder(x.shape[2], x.shape[3])
            spatial_gate = self.spatial_gate_conv(gate)
            channel_gate = self.channel_gate_conv(x.mean(dim=(2, 3), keepdim=True))
            gate = torch.sigmoid(spatial_gate + channel_gate)
            x = self.final_conv(x * gate)
            return x

        elif self.enctype == 'map':
            map = self.relu(self.map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
            spatial_gate = self.spatial_gate_conv(map)
            channel_gate = self.channel_gate_conv(x.mean(dim=(2, 3), keepdim=True))
            gate = torch.sigmoid(spatial_gate + channel_gate)
            x = self.final_conv(x * gate)
            return x

        elif self.enctype == 'bilinear':
            map = create_bilinear_coeff_map_cart_3x3(self.map[:, 0:1], self.map[:, 1:2])
            map = self.relu(map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
            spatial_gate = self.spatial_gate_conv(map)
            channel_gate = self.channel_gate_conv(x.mean(dim=(2, 3), keepdim=True))
            gate = torch.sigmoid(spatial_gate + channel_gate)
            x = self.final_conv(x * gate)
            return x


def create_bilinear_coeff_map_cart_3x3(x_disp, y_disp):
    shape = x_disp.shape
    x_disp = x_disp.reshape(-1)
    y_disp = y_disp.reshape(-1)

    primary_indices = torch.zeros_like(x_disp, dtype=torch.long)
    primary_indices[(x_disp >= 0) & (y_disp >= 0)] = 0  # Quadrant 1
    primary_indices[(x_disp < 0) & (y_disp >= 0)] = 2  # Quadrant 2
    primary_indices[(x_disp < 0) & (y_disp < 0)] = 4  # Quadrant 3
    primary_indices[(x_disp >= 0) & (y_disp < 0)] = 6  # Quadrant 4

    num_directions = 8

    secondary_indices = ((primary_indices + 1) % num_directions).long()
    tertiary_indices = (primary_indices - 1).long()
    tertiary_indices[tertiary_indices < 0] = num_directions - 1

    x_disp = x_disp.abs()
    y_disp = y_disp.abs()

    coeffs = torch.zeros((x_disp.size(0), num_directions + 1), device=x_disp.device)
    batch_indices = torch.arange(x_disp.size(0), device=x_disp.device)

    coeffs[batch_indices, primary_indices] = (x_disp * y_disp)
    coeffs[batch_indices, secondary_indices] = x_disp * (1 - y_disp)
    coeffs[batch_indices, tertiary_indices] = (1 - x_disp) * y_disp
    coeffs[batch_indices, -1] = (1 - x_disp) * (1 - y_disp)

    swappers = (primary_indices == 0) | (primary_indices == 4)

    coeffs[batch_indices[swappers], secondary_indices[swappers]] = (1 - x_disp[swappers]) * y_disp[swappers]
    coeffs[batch_indices[swappers], tertiary_indices[swappers]] = x_disp[swappers] * (1 - y_disp[swappers])

    coeffs = coeffs.view(shape[0], shape[2], shape[3], num_directions + 1).permute(0, 3, 1, 2)
    reorderer = [0, 1, 2, 7, 8, 3, 6, 5, 4]

    return coeffs[:, reorderer, :, :]

class PositionEncoder(nn.Module):
    def __init__(self, size, encoding_layers=3):
        super(PositionEncoder, self).__init__()
        self.encoding_layers = encoding_layers
        self.encoder = nn.Sequential()
        for i in range(encoding_layers):
            if i == 0:
                self.encoder.add_module('linear' + str(i), nn.Linear(2, 64))
            else:
                self.encoder.add_module('linear' + str(i), nn.Linear(64, 64))
            self.encoder.add_module('relu' + str(i), nn.ReLU())

    def forward(self, height, width):
        device = next(self.parameters()).device
        x_coord, y_coord = torch.linspace(-1, 1, int(height), device=device), torch.linspace(-1, 1, int(width), device=device)
        pos = torch.stack(torch.meshgrid((x_coord, y_coord), indexing='ij'), dim=-1).view(-1, 2)
        return self.encoder(pos).view(1, height, width, -1).permute(0, 3, 1, 2)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1]))
        out = torch.matmul(attention, V)
        return out



class InpaintCoarseNet(nn.Module):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintCoarseNet, self).__init__()

        self.res_blocks = residual_blocks

        self.pad1 = nn.ReflectionPad2d(3)
        self.pConv1_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, padding=0)
        self.pnorm1_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        #self.pact1_1 = nn.ReLU()
        self.glu1 = ConvolutionalGLU(in_features=32, out_features=32)


        self.pConv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pnorm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        #self.pact2_1 = nn.ReLU()
        self.glu2 = ConvolutionalGLU(in_features=64, out_features=64)


        self.pConv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.pnorm3_1 = nn.InstanceNorm2d(128, track_running_stats=False)
        #self.pact3_1 = nn.ReLU()
        self.glu3 = ConvolutionalGLU(in_features=128, out_features=128)


        self.pConv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.pnorm4_1 = nn.InstanceNorm2d(256, track_running_stats=False)
        #self.pact4_1 = nn.ReLU()
        self.glu4 = ConvolutionalGLU(in_features=256, out_features=256)


        ##替换部分
        self.block1 = MultiFrequency_Unbiased(in_channels=256, out_channels=256)
        self.block2 = MultiFrequency_Unbiased(in_channels=256, out_channels=256)
        self.block3 = MultiFrequency_Unbiased(in_channels=256, out_channels=256)
        self.block4 = MultiFrequency_Unbiased(in_channels=256, out_channels=256)

        self.coord_gate = UnifiedCoordGate(enc_channels=256, out_channels=256, encoding_layers=2)

        self.conv0_5 = nn.Conv2d(in_channels=256 , out_channels=128, kernel_size=3, padding=1)
        self.norm0_5 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.act0_5 = nn.ReLU()
        self.glu5 = ConvolutionalGLU(in_features=128, out_features=128)


        self.conv1_1 = nn.Conv2d(in_channels=128+128 , out_channels=64, kernel_size=3, padding=1)
        self.norm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act1_1 = nn.ReLU()
        self.glu6 = ConvolutionalGLU(in_features=64, out_features=64)


        self.conv2_1 = nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=3, padding=1)
        self.norm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act2_1 = nn.ReLU()
        self.glu7 = ConvolutionalGLU(in_features=64, out_features=32)


        self.conv3_1 = nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, padding=1)
        self.norm3_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.act3_1 = nn.ReLU()
        self.glu8 = ConvolutionalGLU(in_features=32, out_features=16)


        self.pad2 = nn.ReflectionPad2d(0)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.pad1(x)
        x = self.pConv1_1(x)
        x = self.pnorm1_1(x)
        x = self.glu1(x)
        x1 = x
        x = self.pConv2_1(x)
        x = self.pnorm2_1(x)
        x = self.glu2(x)
        x2 = x
        x = self.pConv3_1(x)
        x = self.pnorm3_1(x)
        x = self.glu3(x)
        x3 = x
        x = self.pConv4_1(x)
        x = self.pnorm4_1(x)
        x = self.glu4(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.coord_gate(x)
        x = self.conv0_5(x)
        x = self.norm0_5(x)
        x = self.glu5(x)
        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x3), dim=1)
        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = self.glu6(x)
        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x2), dim=1)
        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.glu7(x)
        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x1), dim=1)
        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.glu8(x)
        x = self.pad2(x)
        x = self.conv3_2(x)
        x = (torch.tanh(x) + 1) / 2
        x = torch.clamp(x, 0, 1)
        return x





class InpaintRefineNet(nn.Module):
    def __init__(self, residual_blocks=4, init_weights=True):
        super(InpaintRefineNet, self).__init__()

        self.res_blocks = residual_blocks

        self.pad1 = nn.ReflectionPad2d(3)
        self.pConv1_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, padding=0)
        self.pnorm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.pact1_1 = nn.ReLU()

        self.pConv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pnorm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.pact2_1 = nn.ReLU()

        self.pConv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.pnorm3_1 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.pact3_1 = nn.ReLU()

        self.pConv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.pnorm4_1 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.pact4_1 = nn.ReLU()
        ###
        blocks = [MultiFrequency_Unbiased(in_channels=256, out_channels=256)]


        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)


        self.refine_attn = DAFA(256)


        self.middle = nn.Sequential(*blocks)

        self.conv0_5 = nn.Conv2d(in_channels=256 , out_channels=128, kernel_size=3, padding=1)  # +256
        self.norm0_5 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.act0_5 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=128+128 , out_channels=64, kernel_size=3, padding=1)  # +256
        self.norm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act1_1 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(in_channels=64+64 , out_channels=32, kernel_size=3, padding=1)  # +256
        self.norm2_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.act2_1 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(in_channels=32+32 , out_channels=16, kernel_size=3, padding=1)  # +128
        self.norm3_1 = nn.InstanceNorm2d(16, track_running_stats=False)
        self.act3_1 = nn.ReLU()

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7)

    def forward(self, x):
        x0=x[:,:-1,:,:]

        x = self.pad1(x)
        x = self.pConv1_1(x)
        x = self.pnorm1_1(x)
        x = self.act1_1(x)
        x1 = x

        x = self.pConv2_1(x)
        x = self.pnorm2_1(x)
        x = self.pact2_1(x)
        x2 = x

        x = self.pConv3_1(x)
        x = self.pnorm3_1(x)
        x = self.pact3_1(x)
        x3 = x

        x = self.pConv4_1(x)
        x = self.pnorm4_1(x)
        x = self.pact4_1(x)

        x=self.refine_attn([x,x])
        x = self.middle(x)

        x = self.conv0_5(x)
        x = self.norm0_5(x)
        x = self.act0_5(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x3), dim=1)

        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = self.act1_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x2), dim=1)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.act2_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x1), dim=1)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.act3_1(x)

        x = self.pad2(x)
        x = self.conv3_2(x)
        x = (torch.tanh(x) + 1) / 2

        x=torch.clamp(x,0,1)
        x = x + x0
        x=torch.clamp(x,0,1)
        return x

class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=4, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.coarsenet=InpaintCoarseNet(residual_blocks=residual_blocks,init_weights=True)
        self.refinenet=InpaintRefineNet(residual_blocks=residual_blocks,init_weights=True)

        if init_weights:
            self.init_weights()

    def forward(self, x,masks,returnInput2=False,coarseOnly=False):
        x0=x[:,:-1,:,:]

        # x = F.interpolate(x, size=[(int)(x.shape[2] / 2), (int)(x.shape[3] / 2)], mode='bilinear', align_corners=True)

        x1 = self.coarsenet(x)

        newinput_merged = (x1 * masks) + (x0 * (1 - masks))

        channel1mask=masks[:,0,:,:]
        channel1mask=torch.unsqueeze(channel1mask,1)
        newinput_merged_withmask=torch.cat((newinput_merged, channel1mask), dim=1)

        if(coarseOnly):
            x2=x1
            # with torch.no_grad():
            #     x2 = self.refinenet(newinput_merged_withmask)
        else:
            x2=self.refinenet(newinput_merged_withmask)

        if returnInput2:
            return x1,x2,newinput_merged
        else:
            return x1,x2

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            Partial_conv3(dim=in_channels, n_div=2, forward='split_cat'),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            Partial_conv3(dim=in_channels, n_div=2, forward='split_cat'), 
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            Partial_conv3(dim=in_channels, n_div=2, forward='split_cat'),  
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            Partial_conv3(dim=in_channels, n_div=2, forward='split_cat'),  
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            Partial_conv3(dim=in_channels, n_div=2, forward='split_cat'),  
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



class InpaintFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(InpaintFLOPsWrapper, self).__init__()
        self.model = model  

    def forward(self, image):
        fake_mask = torch.randn(image.size(0), 1, image.size(2), image.size(3)).cuda()  
        return self.model(image, fake_mask)  


if __name__ == '__main__':
    model = InpaintGenerator()

    sample_input = torch.randn(1, 4, 256, 256).cuda()
    model = model.cuda()

    summary(model, input_data=(sample_input, torch.randn(1, 1, 256, 256).cuda()))

    flops_wrapper = InpaintFLOPsWrapper(model)

    input_size = (4, 256, 256)
    macs, params = get_model_complexity_info(flops_wrapper, input_size, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print(f"FLOPs: {macs}")
    print(f"参数数量: {params}")
