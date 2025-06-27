import os
import torch
from model import LCNN
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Rafdb')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--resume_net', type=str, default='./results/best.ckpt')
parser.add_argument('--dropout', type=float, default=0.4)  # 添加缺失的 dropout 参数
opts = parser.parse_args()


# 融合卷积层和BN层的函数
def fuse_conv_bn(conv, bn):
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )

    # 计算融合后的权重和偏置
    scale_factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_conv.weight.data = (
        conv.weight * scale_factor.reshape(-1, 1, 1, 1)).clone()

    if conv.bias is not None:
        fused_bias = (conv.bias - bn.running_mean) * scale_factor + bn.bias
    else:
        fused_bias = (-bn.running_mean) * scale_factor + bn.bias
    fused_conv.bias.data = fused_bias.clone()

    return fused_conv


# 融合全连接层和BN层的函数
def fuse_fc_bn(fc, bn):
    fused_fc = torch.nn.Linear(
        fc.in_features,
        fc.out_features,
        bias=True
    )

    scale_factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_fc.weight.data = (fc.weight * scale_factor.reshape(-1, 1)).clone()

    if fc.bias is not None:
        fused_bias = (fc.bias - bn.running_mean) * scale_factor + bn.bias
    else:
        fused_bias = (-bn.running_mean) * scale_factor + bn.bias
    fused_fc.bias.data = fused_bias.clone()

    return fused_fc


# 加载模型
net = LCNN(opts)
ckpt = torch.load(opts.resume_net, map_location=torch.device('cpu'))
net.load_state_dict(ckpt)
net.eval()

# 融合卷积部分的BN层
fused_conv_layers = []
modules = list(net.conv_block.children())
i = 0
while i < len(modules):
    if isinstance(modules[i], torch.nn.Conv2d):
        if i+1 < len(modules) and isinstance(modules[i+1], torch.nn.BatchNorm2d):
            fused_conv = fuse_conv_bn(modules[i], modules[i+1])
            fused_conv_layers.append(fused_conv)
            i += 2  # 跳过BN层
        else:
            fused_conv_layers.append(modules[i])
            i += 1
    else:
        i += 1  # 保留非卷积层（如ReLU, MaxPool）

# 融合全连接部分的BN层
fused_fc_layers = []
modules = list(net.fc.children())
i = 0
while i < len(modules):
    if isinstance(modules[i], torch.nn.Linear):
        if i+1 < len(modules) and isinstance(modules[i+1], torch.nn.BatchNorm1d):
            fused_fc = fuse_fc_bn(modules[i], modules[i+1])
            fused_fc_layers.append(fused_fc)
            i += 2  # 跳过BN层
        else:
            fused_fc_layers.append(modules[i])
            i += 1
    else:
        i += 1  # 保留ReLU和Dropout

# 提取融合后的权重和偏置
conv_weights = []
conv_biases = []
for layer in fused_conv_layers:
    if isinstance(layer, torch.nn.Conv2d):
        conv_weights.append(layer.weight.detach().numpy().astype(np.float32))
        conv_biases.append(layer.bias.detach().numpy().astype(np.float32))

fc_weights = []
fc_biases = []
for layer in fused_fc_layers:
    if isinstance(layer, torch.nn.Linear):
        fc_weights.append(layer.weight.detach().numpy().astype(np.float32))
        fc_biases.append(layer.bias.detach().numpy().astype(np.float32))

# 保存参数（按层拼接）
if not os.path.exists('./params'):
    os.makedirs('./params')

# 保存卷积参数
with open('./params/weight_conv.bin', 'wb') as f:
    for w in conv_weights:
        w.tofile(f)
with open('./params/bias_conv.bin', 'wb') as f:
    for b in conv_biases:
        b.tofile(f)

# 保存全连接参数
with open('./params/weight_fc.bin', 'wb') as f:
    for w in fc_weights:
        w.tofile(f)
with open('./params/bias_fc.bin', 'wb') as f:
    for b in fc_biases:
        b.tofile(f)

print("Parameters saved with fused BN layers.")
