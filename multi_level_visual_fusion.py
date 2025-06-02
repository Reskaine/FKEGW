import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as nms_torch
from functools import partial

from module.utils import MemoryEfficientSwish, Swish
from module.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 out_channel,
                 ratio):
        super(GlobalContextBlock, self).__init__()
        self.out_channel = out_channel
        self.ratio = ratio
        self.planes = int(out_channel * ratio)
        self.conv_mask = nn.Conv2d(out_channel, 1, kernel_size=1)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.out_channel, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(),  # yapf: disable
            nn.Conv2d(self.planes, self.out_channel, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [traditional_non_knowledge, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [traditional_non_knowledge, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [traditional_non_knowledge, 1, H, W]
        context_mask = self.conv_mask(x)
        # [traditional_non_knowledge, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [traditional_non_knowledge, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [traditional_non_knowledge, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        #input_x_t = input_x.permute(0, 2, 1)  # [traditional_non_knowledge, H * W, C, 1]

        context = torch.matmul(input_x, context_mask).squeeze(2)  # [traditional_non_knowledge, C, 1]
        # [traditional_non_knowledge, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [traditional_non_knowledge, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # [traditional_non_knowledge, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.register_buffer('alpha', torch.ones((1, num_channels, 1, 1)))
        self.register_buffer('gamma', torch.zeros((1, num_channels, 1, 1)))
        self.register_buffer('beta', torch.zeros((1, num_channels, 1, 1)))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdims=True).sum(
                3, keepdims=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon

        # Conv layers
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv1_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv2_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)

        # self.p3_ca = GCT(num_channels)
        # self.p2_ca = GCT(num_channels)
        # self.p1_ca = GCT(num_channels)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            # self.p4_1_sa = GlobalContextBlock(num_channels, 0.25)
            # self.p3_1_sa = GlobalContextBlock(num_channels, 0.25)
            # self.p2_1_sa = GlobalContextBlock(num_channels, 0.25)
            # self.p1_1_sa = GlobalContextBlock(num_channels, 0.25)
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p2_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p1_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # self.p2_2_sa = GlobalContextBlock(conv_channels[2], 1.5)
            # self.p3_2_sa = GlobalContextBlock(conv_channels[1], 1.5)
            self.p2_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()

        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()

    def forward(self, inputs):

        p1, p2, p3, p4 = self._forward_fast_attention(inputs)

        return p1, p2, p3, p4

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p1, p2, p3, p4 = inputs

            # p1_in = self.p1_1_sa(self.p1_down_channel(p1))
            # p2_in = self.p2_1_sa(self.p2_down_channel(p2))
            # p3_in = self.p3_1_sa(self.p3_down_channel(p3))
            # p4_in = self.p4_1_sa(self.p4_down_channel(p4))
            p1_in = self.p1_down_channel(p1)
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)

        else:
            # P1_0, P2_0, P3_0, P4_0 and P5_0
            p1_in, p2_in, p3_in, p4_in = inputs

        # P4_0 to P4_2

        # Weights for P3_0 and P4_0 to P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_0 to P3_1 respectively
        p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p4_downsample(p4_in)))

        # Weights for P2_0 and P3_1 to P2_1
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for P2_0 and P3_1 to P2_1 respectively
        p2_up = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p3_downsample(p3_up)))

        # Weights for P1_0 and P2_1 to P1_2
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        # Connections for P1_0 and P2_1 to P1_2 respectively
        p1_out = self.conv1_up(self.swish(weight[0] * p1_in + weight[1] * self.p2_downsample(p2_up)))

        if self.first_time:
            p2_in = self.p2_down_channel_2(p2)
            p3_in = self.p3_down_channel_2(p3)

        # Weights for P2_0, P2_1 and P1_2 to P2_2
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        # Connections for P2_0, P2_1 and P1_2 to P2_2 respectively
        p2_out = self.conv2_down(
            self.swish(weight[0] * p2_in + weight[1] * p2_up + weight[2] * self.p1_upsample(p1_out)))

        # Weights for P3_0, P3_1 and P2_2 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        p3_out = self.conv3_down(
            self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p2_upsample(p2_out)))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P4_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_in + weight[2] * self.p3_upsample(p3_out)))

        return p1_out, p2_out, p3_out, p4_out


class Multi_BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, fpn_cell_repeats):
        super(Multi_BiFPN, self).__init__()

        self.fpn_num_filters = num_channels
        self.fpn_cell_repeats = fpn_cell_repeats
        self.conv_channel_coef = conv_channels

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters,
                    self.conv_channel_coef,
                    True if _ == 0 else False)
              for _ in range(self.fpn_cell_repeats)])

        self.conv4_up_1 = SeparableConvBlock(num_channels, onnx_export=False)
        self.conv3_up_1 = SeparableConvBlock(num_channels, onnx_export=False)
        self.conv2_up_1 = SeparableConvBlock(num_channels, onnx_export=False)
        self.conv1_up_1 = SeparableConvBlock(num_channels, onnx_export=False)

        self.p2_downsample_1 = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample_1 = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample_1 = MaxPool2dStaticSamePadding(3, 2)

        self.p3_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w_relu = nn.ReLU()
        self.p2_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w_relu = nn.ReLU()
        self.p1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w_relu = nn.ReLU()

        self.swish = MemoryEfficientSwish() if not False else Swish()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        p1, p2, p3, p4 = inputs
        p = (p1, p2, p3, p4)
        p1, p2, p3, p4 = self.bifpn(p)

        p3_w = self.p3_w_relu(self.p3_w)
        weight = p3_w / (torch.sum(p3_w, dim=0) + 1e-4)
        p3_result = self.conv3_up_1(self.swish(weight[0] * p3 + weight[1] * self.p4_downsample_1(p4)))

        p2_w = self.p2_w_relu(self.p2_w)
        weight = p2_w / (torch.sum(p2_w, dim=0) + 1e-4)
        p2_result = self.conv2_up_1(self.swish(weight[0] * p2 + weight[1] * self.p3_downsample_1(p3_result)))

        p1_w = self.p1_w_relu(self.p1_w)
        weight = p1_w / (torch.sum(p1_w, dim=0) + 1e-4)
        output = self.conv1_up_1(self.swish(weight[0] * p1 + weight[1] * self.p2_downsample_1(p2_result)))

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        return output