
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


class TestModel(torch.nn.Module):
    def __init__(self, conv_shapes, fc_shapes):
        super(TestModel, self).__init__()

        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(
            3, 32, kernel_size=3, padding=1)  # 输入3通道，输出32通道
        self.conv2 = torch.nn.Conv2d(
            32, 32, kernel_size=3, padding=1)  # 输入32通道，输出32通道
        self.conv3 = torch.nn.Conv2d(
            32, 64, kernel_size=3, padding=1)  # 输入32通道，输出64通道
        self.conv4 = torch.nn.Conv2d(
            64, 128, kernel_size=3, padding=1)  # 输入64通道，输出128通道

        # 全连接层
        # 输入128 * 5 * 5=3200，输出256
        self.fc1 = torch.nn.Linear(128 * 5 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 128)  # 输入256，输出128
        self.fc3 = torch.nn.Linear(128, 7)  # 输入128，输出7

        # 其他层
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 第一卷积块
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # 第二卷积块
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # 第三卷积块
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        # 全连接块
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 定义网络结构参数
conv_shapes = [
    (32, 3, 3, 3),   # conv1: 输出通道32, 输入通道3, 卷积核3x3
    (32, 32, 3, 3),  # conv2: 输出通道32, 输入通道32, 卷积核3x3
    (64, 32, 3, 3),  # conv3: 输出通道64, 输入通道32, 卷积核3x3
    (128, 64, 3, 3)  # conv4: 输出通道128, 输入通道64, 卷积核3x3
]

fc_shapes = [
    (256, 128 * 5 * 5),   # fc1: 输出256, 输入128 * 5 * 5
    (128, 256),           # fc2: 输出128, 输入256
    (7, 128)              # fc3: 输出7, 输入128
]

# 创建测试模型
test_model = TestModel(conv_shapes, fc_shapes)
test_model.eval()

# 加载卷积权重
with open('./params/weight_conv.bin', 'rb') as f:
    weights_data = np.frombuffer(f.read(), dtype=np.float32)

# 按层加载参数
conv_layers = [test_model.conv1, test_model.conv2,
               test_model.conv3, test_model.conv4]
start_idx = 0
for i, layer in enumerate(conv_layers):
    shape = conv_shapes[i]
    size = np.prod(shape)
    w = weights_data[start_idx:start_idx+size].reshape(shape)
    layer.weight.data = torch.tensor(w).float()
    start_idx += size

# 加载卷积偏置
with open('./params/bias_conv.bin', 'rb') as f:
    biases_data = np.frombuffer(f.read(), dtype=np.float32)

start_idx = 0
for i, layer in enumerate(conv_layers):
    shape = conv_shapes[i]
    size = shape[0]
    b = biases_data[start_idx:start_idx+size]
    layer.bias.data = torch.tensor(b).float()
    start_idx += size

# 加载全连接权重
with open('./params/weight_fc.bin', 'rb') as f:
    weights_data = np.frombuffer(f.read(), dtype=np.float32)

fc_layers = [test_model.fc1, test_model.fc2, test_model.fc3]
start_idx = 0
for i, layer in enumerate(fc_layers):
    shape = fc_shapes[i]
    size = np.prod(shape)
    w = weights_data[start_idx:start_idx+size].reshape(shape)
    layer.weight.data = torch.tensor(w).float()
    start_idx += size

# 加载全连接偏置
with open('./params/bias_fc.bin', 'rb') as f:
    biases_data = np.frombuffer(f.read(), dtype=np.float32)

start_idx = 0
for i, layer in enumerate(fc_layers):
    shape = fc_shapes[i]
    size = shape[0]
    b = biases_data[start_idx:start_idx+size]
    layer.bias.data = torch.tensor(b).float()
    start_idx += size

# 图像预处理
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
img = Image.open('./data/image.bmp')
img_tensor = transform(img).unsqueeze(0)

# 原始模型输出
with torch.no_grad():
    logits_original = net(img_tensor)

# 测试模型输出
with torch.no_grad():
    logits_test = test_model(img_tensor)

error = np.mean(abs(logits_original.detach().numpy() -
                logits_test.detach().numpy()))
print('Logits: {}'.format(logits_test.detach().numpy()))
print('Error Logits: {:.10f}'.format(error))

# 输出预测结果
_, predict = torch.max(logits_test, 1)
label2emotion = ['surprise', 'fear', 'disgust',
                 'happiness', 'sadness', 'anger', 'neutral']
print('Prediction: {}'.format(label2emotion[predict.item()]))
