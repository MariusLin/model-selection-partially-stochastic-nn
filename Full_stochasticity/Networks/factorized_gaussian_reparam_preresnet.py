import torch
import torch.nn as nn
import torch.nn.functional as F


from Partial_stochasticity.Layers.factorized_gaussian_reparam_conv import FactorizedGaussianConv2dReparameterization
from Partial_stochasticity.Layers.factorized_gaussian_linear_reparam import FactorizedGaussianLinearReparameterization

"""
This is an implementation of the PreResNet from He et al. 2016
All the parameters have a reparameterized Gaussian prior on it 
where the standard deviation is factorized using DWF
"""
def conv3x3(in_planes, out_planes, stride=1, bias = False, 
            W_std=None, b_std=None, prior_per="layer",
            scaled_variance=True, device = "cpu", D= 3):
    "3x3 convolution with padding"
    return FactorizedGaussianConv2dReparameterization(
        in_planes, out_planes, D= D, kernel_size=3, stride=stride, W_std = W_std, b_std = b_std, prior_per=prior_per,
        padding=1, bias=bias, scaled_variance=scaled_variance, device = device)


def init_norm_layer(inplanes, norm_layer):
    assert norm_layer in ['batchnorm', 'frn', None]
    if norm_layer == 'batchnorm':
        return nn.BatchNorm2d(inplanes, eps=0, momentum=None, affine=False,
                              track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer='batchnorm', W_std=None, b_std=None, bias = False,
                 prior_per="layer", scaled_variance=True, device = "cpu", D= 3):
        super(BasicBlock, self).__init__()
        # An additional paramter is the factorization depth D
        prior_params = {'W_std': W_std, 'b_std': b_std,
                        'prior_per': prior_per,
                        'scaled_variance': scaled_variance,
                        'D': D, 'device': device, 'bias':bias}
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, **prior_params)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        self.conv2 = conv3x3(planes, planes, **prior_params)
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
    
    def get_W_std_masks(self):
        """
        This returns the mask with the deterministic and stochastic weights
        It is of the follwing shape:
        [mask conv1, mask conv2, [mask for downsample layer]]
        """
        W_std_masks = []
        W_std_masks.append(self.conv1.get_W_std_mask())
        W_std_masks.append(self.conv2.get_W_std_mask())
        if self.downsample is not None:
            downsample_list = []
            for l in self.downsample:
                downsample_list.append(l.get_W_std_mask())
            W_std_masks.append(downsample_list)
        return W_std_masks

    def get_b_std_masks(self):
        """
        This returns the mask with the deterministic and stochastic bias parameters
        It is of the follwing shape:
        [mask conv1, mask conv2, [mask for downsample layer]]
        """
        b_std_masks = []
        b_std_masks.append(self.conv1.get_b_std_mask())
        b_std_masks.append(self.conv2.get_b_std_mask())
        if self.downsample is not None:
            downsample_list = []
            for l in self.downsample:
                if l.bias:
                    downsample_list.append(l.get_b_std_mask())
            b_std_masks.append(downsample_list)
        return b_std_masks
    
    def get_W_std(self):
        """
        This returns the standard deviations of the weights
        It is of the following form: [conv1, conv2, downsample]
        """
        W_std = []
        W_std.append(self.conv1.get_W_std())
        W_std.append(self.conv2.get_W_std())
        if self.downsample is not None:
            for l in self.downsample:
                W_std.append(l.get_W_std())
        return W_std
    
    def get_b_std(self):
        """
        This returns the standard deviations of the bias parameters
        It is of the following form: [conv1, conv2, downsample]
        """
        b_std = []
        if self.conv1.bias:
            b_std.append(self.conv1.get_b_std())
        if self.conv2.bias:
           b_std.append(self.conv2.get_b_std())
        if self.downsample is not None:
            for l in self.downsample:
                if l.bias:
                    b_std.append(l.get_b_std())
        return b_std

    def get_num_pruned_W_std(self):
        """
        Here the number of pruned standard deviations of the weights is counted
        """
        W_std_list = self.get_W_std()
        num_pruned = 0
        for W_std in W_std_list:
            num_pruned += (W_std == 0).sum().item()
        return num_pruned
    
    def get_num_pruned_b_std(self):
        """
        Here the number of pruned standard deviations of the bias parameters is counted
        """
        b_std_list = self.get_b_std()
        num_pruned = 0
        for b_std in b_std_list:
            num_pruned += (b_std == 0).sum().item()
        return num_pruned


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer='batchnorm', W_std=None, b_std=None, bias = False,
                 prior_per="layer", scaled_variance=True, device = "cpu", D=3):
        super(Bottleneck, self).__init__()
        prior_params = {'W_std': W_std, 'b_std': b_std,
                        'prior_per': prior_per,
                        'scaled_variance': scaled_variance,
                        'D':D , 'device': device}
        self.bias = bias
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        self.conv1 = FactorizedGaussianConv2dReparameterization(
            inplanes, planes, kernel_size=1, bias=self.bias, **prior_params)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        self.conv2 = FactorizedGaussianConv2dReparameterization(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=self.bias, **prior_params)
        self.norm3 = init_norm_layer(inplanes, norm_layer)
        self.conv3 = FactorizedGaussianConv2dReparameterization(
            planes, planes * 4, kernel_size=1, bias=self.bias, **prior_params)
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
    
    def get_W_std_masks(self):
        """
        This returns the mask with the deterministic and stochastic weights
        It is of the follwing shape:
        [mask conv1, mask conv2, mask conv3, [mask for downsample layer]]
        """
        W_std_masks = []
        W_std_masks.append(self.conv1.get_W_std_mask())
        W_std_masks.append(self.conv2.get_W_std_mask())
        W_std_masks.append(self.conv3.get_W_std_mask())
        if self.downsample is not None:
            downsample_list = []
            for l in self.downsample:
                downsample_list.append(l.get_W_std_mask())
            W_std_masks.append(downsample_list)
        return W_std_masks

    def get_b_std_masks(self):
        """
        This returns the mask with the deterministic and stochastic bias parameters
        It is of the follwing shape:
        [mask conv1, mask conv2, mask conv3, [mask for downsample layer]]
        """
        b_std_masks = []
        b_std_masks.append(self.conv1.get_b_std_mask())
        b_std_masks.append(self.conv2.get_b_std_mask())
        b_std_masks.append(self.conv3.get_b_std_mask())
        if self.downsample is not None:
            downsample_list = []
            for l in self.downsample:
                if l.bias:
                    downsample_list.append(l.get_b_std_mask())
            b_std_masks.append(b_std_masks)
        return b_std_masks
    
    def get_W_std(self):
        """
        This returns the standard deviations of the weights
        It is of the following form: [conv1, conv2, conv3, downsample]
        """
        W_std = []
        W_std.append(self.conv1.get_W_std())
        W_std.append(self.conv2.get_W_std())
        W_std.append(self.conv3.get_W_std())
        if self.downsample is not None:
            for l in self.downsample:
                W_std.append(l.get_W_std())
        return W_std
    
    def get_b_std(self):
        """
        This returns the standard deviations of the bias parameters
        It is of the following form: [conv1, conv2, conv3, downsample]
        """
        b_std = []
        if self.conv1.bias:
            b_std.append(self.conv1.get_b_std())
        if self.conv2.bias:
           b_std.append(self.conv2.get_b_std())
        if self.conv3.bias:
            b_std.append(self.conv3.get_b_std())
        if self.downsample is not None:
            for l in self.downsample:
                if l.bias:
                    b_std.append(l.get_b_std())
        return b_std

    def get_num_pruned_W_std(self):
        """
        Here the number of pruned standard deviations of the weights is counted
        """
        W_std_list = self.get_W_std()
        num_pruned = 0
        for W_std in W_std_list:
            num_pruned += (W_std == 0).sum().item()
        return num_pruned
    
    def get_num_pruned_b_std(self):
        """
        Here the number of pruned standard deviations of the bias parameters is counted
        """
        b_std_list = self.get_b_std()
        num_pruned = 0
        for b_std in b_std_list:
            num_pruned += (b_std == 0).sum().item()
        return num_pruned


class FactorizedGaussianPreResNetReparameterization(nn.Module):
    def __init__(self, depth, num_classes=10, block_name='BasicBlock',
                 norm_layer='batchnorm', W_std=None, b_std=None, bias = False,
                 prior_per="layer", scaled_variance=True, device = "cpu", D=3):
        super(FactorizedGaussianPreResNetReparameterization, self).__init__()
        self.prior_params = {'W_std': W_std, 'b_std': b_std,
                             'prior_per': prior_per, 'D': D,
                             'scaled_variance': scaled_variance,
                             'device': device}
        self.scaled_variance = scaled_variance
        self.D = D
        self.device = device
        self.bias = bias
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
        self.conv1 = FactorizedGaussianConv2dReparameterization(
            3, 16, kernel_size=3, padding=1, bias=self.bias, **self.prior_params)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.norm = init_norm_layer(64 * block.expansion, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = FactorizedGaussianLinearReparameterization(
            64 * block.expansion, num_classes, **self.prior_params)
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.output_layer = self.fc

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                FactorizedGaussianConv2dReparameterization(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=stride, bias=self.bias, **self.prior_params),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            **self.prior_params))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **self.prior_params))

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
        predictions = self.forward(x)

        # Take exponential because of using log_softmax in the last layer of
        # the model.
        predictions = torch.exp(predictions)

        return predictions

    def sample_functions(self, X, n_samples, log_softmax=False):
        samples = []
        for i in range(n_samples):
            sample = self.forward(X, log_softmax)
            samples.append(sample.cpu())

        samples = torch.stack(samples, dim=0).transpose(0, 1).to(X.device)
        return samples
    
    def get_det_masks (self):
        """
        This returns the masks of the following shape
        (weight masks, bias masks)
        Each mask is itself of the following form:
        [[Mask for the input layer],[[Masks layer 1 (possibly with mask for the downstream layer {only in first block})]], 
        ..., [Mask for the last layer (possibly with mask for the downstream layer {only in first block})], 
        [Mask for the output layer]]
        """
        layer1_list_weight = []
        layer1_list_bias = []
        for block in self.layer1:
            layer1_list_weight.append(block.get_W_std_masks())
            if self.bias:
                layer1_list_bias.append(block.get_b_std_masks())
        layer2_list_weight = []
        layer2_list_bias = []
        for block in self.layer2:
            layer2_list_weight.append(block.get_W_std_masks())
            if self.bias:
                layer2_list_bias.append(block.get_b_std_masks())
        layer3_list_weight = []
        layer3_list_bias = []
        for block in self.layer3:
            layer3_list_weight.append(block.get_W_std_masks())
            if self.bias:
                layer3_list_bias.append(block.get_b_std_masks())
        det_mask_weights = [[self.conv1.get_W_std_mask()],layer1_list_weight, 
                            layer2_list_weight, layer3_list_weight]
        det_mask_weights.append([self.fc.get_W_std_mask()])
        if self.bias:
            det_mask_bias = [[self.conv1.get_b_std_mask()], layer1_list_bias, 
                             layer2_list_bias, layer3_list_bias]
            output_bias_mask = []
            output_bias_mask.append(self.fc.get_b_std_mask())

            return det_mask_weights, det_mask_bias, output_bias_mask
        else:
            output_bias_mask = []
            output_bias_mask.append(self.fc.get_b_std_mask())
            return det_mask_weights, None, output_bias_mask
    

    def to(self, device):
        self.device = torch.device(device)
        super().to(device)
        return self 