import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st
from torch.nn import init
import argparse
import numpy as np
import glob
from PIL import Image

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def res2net50():
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)
    return model

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1), grid, grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy  

class FourDirectionalExpand(nn.Module):
    def __init__(self, in_Channel):
        super(FourDirectionalExpand, self).__init__()
        self.in_Channel = in_Channel
        self.adjust_channels = nn.Conv2d(in_Channel, in_Channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        B, _, H, W = x.shape
        
        # 上下展开
        up = torch.cat([x[:, :, i, :].reshape(B, -1) for i in range(H)], dim=1)
        
        # 左右展开
        left = torch.cat([x[:, :, :, i].reshape(B, -1) for i in range(W)], dim=1)
        
        return up, left

    def restore_from_directional(self, H, W, up, left, ori):
        B,_ = up.shape
        
        # 恢复上下方向
        up_restored = torch.stack([up[:, i*W:(i+1)*W] for i in range(H)], dim=2)
        
        # 恢复左右方向
        left_restored = torch.stack([left[:, i*H:(i+1)*H] for i in range(W)], dim=2)
        
        combined = up_restored + left_restored
        combined = combined.unsqueeze(1).expand(B, self.in_Channel, H, W) + ori
        res = self.adjust_channels(combined)
        return torch.sigmoid(res)
    
class KANlayer(nn.Module):
    def __init__(self, Length, in_Channel):
        super(KANlayer, self).__init__()
        self.length = Length
        self.expand = FourDirectionalExpand(in_Channel)
        self.conv = nn.Conv2d(in_Channel, 1, kernel_size=1, stride=1, padding=0)
        in_features = out_features = self.length * self.length
        self.kan_up = KANLinear(in_features, out_features)
        self.kan_left = KANLinear(in_features, out_features)
        
    def forward(self, x):
        ori = x
        x = self.conv(x)
        up, left = self.expand(x)
        
        up = self.kan_up(up)
        left = self.kan_left(left)
        
        res = self.expand.restore_from_directional(self.length, self.length, up, left, ori)
        
        return res

class KANblock(nn.Module):
    def __init__(self, length, in_channel, num_layers):
        super(KANblock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = KANlayer(length, in_channel)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dropout(x)

        return x

class KMC_UNet(nn.Module):
    def __init__(self, length):
        super(KMC_UNet, self).__init__()
        self.resnet = res2net50()
        self.l1 = length / 4
        self.kan1_1 = KANblock(int(self.l1 / 2), 64, 1)
        self.kan1_2 = KANblock(int(self.l1 / 4), 64, 1)
        self.kan1_3 = KANblock(int(self.l1), 64, 1)
        
        self.l2 = self.l1 / 2
        self.kan2_1 = KANblock(int(self.l2 / 2), 128, 1)
        self.kan2_2 = KANblock(int(self.l2 / 4), 128, 1)
        self.kan2_3 = KANblock(int(self.l2), 128, 1)
        
        self.l3 = self.l2 / 2
        self.kan3_1 = KANblock(int(self.l3 / 2), 256, 1)
        self.kan3_2 = KANblock(int(self.l3 / 4), 256, 1)
        self.kan3_3 = KANblock(int(self.l3), 256, 1)
        
        self.l4 = self.l3 / 2
        self.kan4_1 = KANblock(int(self.l4 / 2), 512, 1)
        self.kan4_2 = KANblock(int(self.l4 / 4), 512, 1)
        self.kan4_3 = KANblock(int(self.l4), 512, 1)
        
        self.uplayer1 = DoubleConv(1024 + 512, 512)
        self.uplayer2 = DoubleConv(512 + 256, 256)
        self.uplayer3 = DoubleConv(256 + 128, 128)
        self.uplayer4 = DoubleConv(128 + 64, 64)
        self.finallayer = nn.Conv2d(64, 1, 1)
        
    def KAN_patches(self, x, patch_size, blocks):
        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            "Image dimensions must be divisible by the patch size."
        
        # 1. 分割图片为patch
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)

        # 2. 处理每个patch
        processed_patches = []
        for i in range(patches.shape[2]):
            patch = patches[:, :, i, :, :]
            patch = blocks(patch)  # 对patch进行处理
            processed_patches.append(patch)

        # 将处理后的patch堆叠
        processed_patches = torch.stack(processed_patches, dim=2)

        # 获取新的通道数和patch大小
        new_C = processed_patches.shape[1]  # 例如 64
        new_patch_size = processed_patches.shape[-1]  # 例如 8

        # 3. 处理形状变换（注意新的patch大小）
        processed_patches = processed_patches.view(B, new_C, H // patch_size, W // patch_size, new_patch_size, new_patch_size)

        # 根据新的patch大小还原原图大小
        processed_patches = processed_patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, new_C, H * new_patch_size // patch_size, W * new_patch_size // patch_size)

        return processed_patches
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        x1 = self.resnet.conv1(x)       
        x1 = self.resnet.bn1(x1)         
        x1 = self.resnet.relu(x1)        
        x1 = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x1)
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)
        x5 = self.resnet.layer4(x4)
        
        x4 = x4 * torch.sigmoid(self.kan4_3(x4) + self.KAN_patches(x4, int(self.l4 / 2), self.kan4_1) + self.KAN_patches(x4, int(self.l4 / 4), self.kan4_2))
        x3 = x3 * torch.sigmoid(self.kan3_3(x3) + self.KAN_patches(x3, int(self.l3 / 2), self.kan3_1) + self.KAN_patches(x3, int(self.l3 / 4), self.kan3_2))
        x2 = x2 * torch.sigmoid(self.kan2_3(x2) + self.KAN_patches(x2, int(self.l2 / 2), self.kan2_1) + self.KAN_patches(x2, int(self.l2 / 4), self.kan2_2))
        x1 = x1 * torch.sigmoid(self.kan1_3(x1) + self.KAN_patches(x1, int(self.l1 / 2), self.kan1_1) + self.KAN_patches(x1, int(self.l1 / 4), self.kan1_2))
        
        x6 = nn.Upsample(scale_factor = 2, mode ="bilinear")(x5)
        x6 = torch.cat([x6, x4], dim = 1)
        x6 = self.uplayer1(x6)
        
        x7 = nn.Upsample(scale_factor = 2, mode ="bilinear")(x6)
        x7 = torch.cat([x7, x3], dim = 1)
        x7 = self.uplayer2(x7)
        
        x8 = nn.Upsample(scale_factor = 2, mode ="bilinear")(x7)
        x8 = torch.cat([x8, x2], dim = 1)
        x8 = self.uplayer3(x8)
        
        x9 = nn.Upsample(scale_factor = 2, mode ="bilinear")(x8)
        x9 = torch.cat([x9, x1], dim = 1)
        x9 = self.uplayer4(x9)
        x9 = self.finallayer(x9)
        
        res = torch.sigmoid(F.interpolate(x9, scale_factor=(4,4), mode='bilinear', align_corners=False))
        
        return res

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


net = KMC_UNet(256).to(device)
opt = optim.Adam(net.parameters())#激活函数
loss_fun = BceDiceLoss()#损失函数

