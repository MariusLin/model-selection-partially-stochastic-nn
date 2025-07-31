import torch
import torch.nn as nn
import torch.nn.functional as F

from Partial_stochasticity.Layers.masked_conv import MaskedConv2d
from Partial_stochasticity.Layers.masked_linear import MaskedLinear
from Partial_stochasticity.Layers.factorized_linear import FactorizedLinear

"""
It is an implementation of the PreResNet of He et al. 2016
It takes two additional parameters: the deterministic masks. They are of the following form:
[[Mask for the input layer],[[Masks layer 1 (possibly with mask for the downstream layer {only in first block})]], 
..., [Mask for the last layer (possibly with mask for the downstream layer {only in first block})], 
[Mask for the output layer]]
"""
def conv3x3(in_planes, out_planes, stride=1, D=3, W_det_mask = None, b_det_mask = None, 
            bias = False, scaled_variance=True, device = "cpu"):
    "3x3 convolution with padding"
    return MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, W_det_mask= W_det_mask,D = D, 
                        b_det_mask=b_det_mask, padding=1, bias=bias, scaled_variance=scaled_variance, device = device)


def init_norm_layer(inplanes, norm_layer, **kwargs):
    assert norm_layer in ['batchnorm',  None]
    if norm_layer == 'batchnorm':
        return nn.BatchNorm2d(inplanes, eps=0, momentum=None, affine=False,
                              track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, W_det_masks = None, b_det_masks = None,
                 norm_layer='batchnorm', scaled_variance=True, bias = False, device = "cpu", D=3):
        super(BasicBlock, self).__init__()
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        if b_det_masks:
            self.conv1 = conv3x3(inplanes, planes, stride, D= D, b_det_mask=b_det_masks[0], W_det_mask=W_det_masks[0],
                bias = bias, scaled_variance=scaled_variance, device = device)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride, D = D, W_det_mask=W_det_masks[0],
                bias = bias, scaled_variance=scaled_variance, device = device)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        if b_det_masks:
            self.conv2 = conv3x3(planes, planes, D=D, W_det_mask=W_det_masks[1], b_det_mask=b_det_masks[1], bias = bias,
                                scaled_variance=scaled_variance, device = device)
        else:
            self.conv2 = conv3x3(planes, planes,D=D, W_det_mask=W_det_masks[1], bias = bias,
                                scaled_variance=scaled_variance, device = device)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
    def get_nums_pruned_W(self):
        """
        Returns the number of pruned weights
        """
        num_pruned_stoch = 0
        num_pruned_det =  0
        num_pruned_s_c1, num_pruned_d_c1 = self.conv1.get_nums_pruned_W()
        num_pruned_stoch += num_pruned_s_c1
        num_pruned_det += num_pruned_d_c1
        num_pruned_s_c2, num_pruned_d_c2 = self.conv2.get_nums_pruned_W()
        num_pruned_stoch += num_pruned_s_c2
        num_pruned_det += num_pruned_d_c2
        if self.downsample:
            for l in self.downsample:
                num_pruned_s, num_pruned_d = l.get_nums_pruned_W()
                num_pruned_stoch += num_pruned_s
                num_pruned_det += num_pruned_d
        return num_pruned_stoch, num_pruned_det


    def get_nums_pruned_b(self):
        """
        Returns the number of pruned bias parameters
        """
        num_pruned_stoch = 0
        num_pruned_det =  0
        num_pruned_s_c1, num_pruned_d_c1 = self.conv1.get_nums_pruned_b()
        num_pruned_stoch += num_pruned_s_c1
        num_pruned_det += num_pruned_d_c1
        num_pruned_s_c2, num_pruned_d_c2 = self.conv2.get_nums_pruned_b()
        num_pruned_stoch += num_pruned_s_c2
        num_pruned_det += num_pruned_d_c2
        if self.downsample:
            for l in self.downsample:
                num_pruned_s, num_pruned_d = l.get_nums_pruned_b()
                num_pruned_stoch += num_pruned_s
                num_pruned_det += num_pruned_d
        return num_pruned_stoch, num_pruned_det


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, W_det_masks = None, b_det_masks = None,
                 norm_layer='batchnorm', scaled_variance=True, bias = False, device = "cpu", D=3):
        super(Bottleneck, self).__init__()
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        if b_det_masks:
            self.conv1 = MaskedConv2d(inplanes, planes, D = D, kernel_size=1, bias=bias, b_det_mask= b_det_masks[0],
                W_det_mask= W_det_masks[0], scaled_variance=scaled_variance, device = device)
        else:
            self.conv1 = MaskedConv2d(inplanes, planes, D= D, kernel_size=1, bias=bias, W_det_mask=W_det_masks[0],
                            scaled_variance=scaled_variance, device = device)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        if b_det_masks:
            self.conv2 = MaskedConv2d(planes, planes, D=D, kernel_size=3, stride=stride,
            b_det_mask=b_det_masks[1], W_det_mask=W_det_masks[1], padding=1, bias=bias,
            scaled_variance=scaled_variance, device = device)
        else:
            self.conv2 = MaskedConv2d(planes, planes, D=D, kernel_size=3, stride=stride,
                                padding=1, bias=bias, W_det_mask = W_det_masks[1],
                                scaled_variance=scaled_variance, device = device)
        self.norm3 = init_norm_layer(inplanes, norm_layer)
        if b_det_masks:
            self.conv3 = MaskedConv2d(planes, planes * 4, D=D, kernel_size=1, bias=bias, b_det_mask= b_det_masks[2],
            W_det_mask=W_det_masks[2], scaled_variance=scaled_variance, device = device)
        else:
            self.conv3 = MaskedConv2d(planes, planes * 4, D=D, kernel_size=1, bias=bias,
            W_det_mask=W_det_masks[2], scaled_variance=scaled_variance, device = device)    
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
    def get_nums_pruned_W(self):
        """
        Returns the number of pruned weights
        """
        num_pruned_stoch = 0
        num_pruned_det =  0
        num_pruned_s_c1, num_pruned_d_c1 = self.conv1.get_nums_pruned_W()
        num_pruned_stoch += num_pruned_s_c1
        num_pruned_det += num_pruned_d_c1
        num_pruned_s_c2, num_pruned_d_c2 = self.conv2.get_nums_pruned_W()
        num_pruned_stoch += num_pruned_s_c2
        num_pruned_det += num_pruned_d_c2
        num_pruned_s_c3, num_pruned_d_c3 = self.conv3.get_nums_pruned_W()
        num_pruned_stoch += num_pruned_s_c3
        num_pruned_det += num_pruned_d_c3
        if self.downsample:
            for l in self.downsample:
                num_pruned_s, num_pruned_d = l.get_nums_pruned_W()
                num_pruned_stoch += num_pruned_s
                num_pruned_det += num_pruned_d
        return num_pruned_stoch, num_pruned_det


    def get_nums_pruned_b(self):
        """
        Returns the number of pruned bias parameters
        """
        num_pruned_stoch = 0
        num_pruned_det =  0
        num_pruned_s_c1, num_pruned_d_c1 = self.conv1.get_nums_pruned_b()
        num_pruned_stoch += num_pruned_s_c1
        num_pruned_det += num_pruned_d_c1
        num_pruned_s_c2, num_pruned_d_c2 = self.conv2.get_nums_pruned_b()
        num_pruned_stoch += num_pruned_s_c2
        num_pruned_det += num_pruned_d_c2
        num_pruned_s_c3, num_pruned_d_c3 = self.conv3.get_nums_pruned_b()
        num_pruned_stoch += num_pruned_s_c3
        num_pruned_det += num_pruned_d_c3
        if self.downsample:
            for l in self.downsample:
                num_pruned_s, num_pruned_d = l.get_nums_pruned_b()
                num_pruned_stoch += num_pruned_s
                num_pruned_det += num_pruned_d
        return num_pruned_stoch, num_pruned_det
    


class MaskedPreResNetNF(nn.Module):
    def __init__(self, depth, weight_masks, bias_masks, bias_mask_out, D=3, num_classes=10, block_name='BasicBlock',
                 norm_layer='batchnorm', scaled_variance=True, device = "cpu", prior_W_std = 1, prior_b_std = 1):
        super(MaskedPreResNetNF, self).__init__()
        self.scaled_variance = scaled_variance
        self.device = device
        self.D = D
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        if bias_masks:
            self.conv1 = MaskedConv2d(3, 16, kernel_size=3, D=D, padding=1, b_det_mask=bias_masks[0][0],
               W_det_mask=weight_masks[0][0], bias=False, scaled_variance=scaled_variance, device = self.device)
        else:
            self.conv1 = MaskedConv2d(3, 16, kernel_size=3, D=D, padding=1, W_det_mask=weight_masks[0][0],
                    bias=False, scaled_variance=scaled_variance, device = self.device)
        if bias_masks:
            self.layer1 = self._make_layer(block, 16, n, weight_masks[1], D, bias_masks[1])
            self.layer2 = self._make_layer(block, 32, n, weight_masks[2], D, bias_masks[2], stride=2)
            self.layer3 = self._make_layer(block, 64, n, weight_masks[3], D, bias_masks[3], stride=2)
        else:
            self.layer1 = self._make_layer(block, 16, n, weight_masks[1], D)
            self.layer2 = self._make_layer(block, 32, n, weight_masks[2], D, stride=2)
            self.layer3 = self._make_layer(block, 64, n, weight_masks[3], D, stride=2)
        self.norm = init_norm_layer(64 * block.expansion, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = MaskedLinear(64 * block.expansion, num_classes, b_det_mask=bias_mask_out[0], D=D,
                    W_det_mask = weight_masks[-1], scaled_variance=scaled_variance,
                    prior_b_std= prior_b_std, prior_W_std=prior_W_std, device = self.device)
        
        self.layers = [self.conv1, self.layer1, self.layer2, self.layer3]
        self.output_layer = self.fc

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, MaskedLinear) or isinstance(m, MaskedConv2d):
                m.reset_parameters()

    def _make_layer(self, block, planes, blocks, W_det_masks, D=3, b_det_masks = None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if b_det_masks:
                downsample = nn.Sequential(
                    MaskedConv2d(self.inplanes, planes * block.expansion, D= D,
                        kernel_size=1, stride=stride, bias=False, W_det_mask=W_det_masks[0][-1][0], 
                        b_det_mask=b_det_masks[0][-1][0], scaled_variance=self.scaled_variance, device= self.device),
                )
            else:
                downsample = nn.Sequential(
                    MaskedConv2d(self.inplanes, planes * block.expansion, D= D,
                        kernel_size=1, stride=stride, bias=False, W_det_mask=W_det_masks[0][-1][0], b_det_mask=None,
                        scaled_variance=self.scaled_variance, device= self.device),
                )

        layers = []
        if b_det_masks:
            layers.append(block(self.inplanes, planes, stride, downsample, D= D, scaled_variance=self.scaled_variance, 
                                W_det_masks = W_det_masks[0], b_det_masks = b_det_masks[0], device = self.device))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, D= D, scaled_variance=self.scaled_variance, 
                                    W_det_masks = W_det_masks[i], b_det_masks = b_det_masks[i], device = self.device))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, D= D, scaled_variance=self.scaled_variance, 
                                W_det_masks = W_det_masks[0], device = self.device))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, D= D, scaled_variance=self.scaled_variance, 
                                    W_det_masks = W_det_masks[i], device = self.device))

        return nn.Sequential(*layers)

    def forward(self, x, log_softmax=False):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.norm(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if log_softmax:
            x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        self.eval()
        predictions = self.forward(x, log_softmax=True)

        predictions = torch.exp(predictions)

        return predictions

    def get_nums_pruned(self):
        """
        Returns the number of pruned parameters
        """
        num_pruned_stoch = 0
        num_pruned_det =  0
        for block in self.layers + [self.output_layer]:
            if isinstance(block, nn.Sequential):
                for layer in block:
                    nums_pruned_W = layer.get_nums_pruned_W()
                    nums_pruned_b = layer.get_nums_pruned_b()
                    num_pruned_stoch += nums_pruned_W[0]
                    num_pruned_stoch += nums_pruned_b[0]
                    num_pruned_det += nums_pruned_W[1]
                    num_pruned_det += nums_pruned_b[1]
            else:
                nums_pruned_W = block.get_nums_pruned_W()
                nums_pruned_b = block.get_nums_pruned_b()
                num_pruned_stoch += nums_pruned_W[0]
                num_pruned_stoch += nums_pruned_b[0]
                num_pruned_det += nums_pruned_W[1]
                num_pruned_det += nums_pruned_b[1]
        return num_pruned_stoch, num_pruned_det
