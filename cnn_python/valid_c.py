
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import struct

# 定义网络结构参数
conv_shapes = [
    (32, 3, 3, 3),   # conv1
    (32, 32, 3, 3),   # conv2
    (64, 32, 3, 3),   # conv3
    (128, 64, 3, 3)   # conv4
]
fc_shapes = [
    (256, 128 * 5 * 5),   # fc1
    (128, 256),       # fc2
    (7, 128)          # fc3
]


def read_bin_as_floats(filename):
    floats = []
    with open(filename, 'rb') as file:
        data = file.read()
        float_size = struct.calcsize('f')

        if len(data) % float_size != 0:
            print("File size is not a multiple of float size. Data might be corrupted.")
            return []
        for i in range(0, len(data), float_size):
            float_value = struct.unpack('f', data[i:i + float_size])[0]
            floats.append(float_value)
    return np.array(floats)


# 读取参数文件
weight_conv_all = read_bin_as_floats('./params/weight_conv.bin')
bias_conv_all = read_bin_as_floats('./params/bias_conv.bin')
weight_fc_all = read_bin_as_floats('./params/weight_fc.bin')
bias_fc_all = read_bin_as_floats('./params/bias_fc.bin')

# 按层分割参数
conv_weights = []
conv_biases = []
start_idx = 0
for shape in conv_shapes:
    size = np.prod(shape)
    w = weight_conv_all[start_idx:start_idx+size].reshape(shape)
    conv_weights.append(w)
    start_idx += size

start_idx = 0
for shape in conv_shapes:
    size = shape[0]
    b = bias_conv_all[start_idx:start_idx+size]
    conv_biases.append(b)
    start_idx += size

fc_weights = []
fc_biases = []
start_idx = 0
for shape in fc_shapes:
    size = np.prod(shape)
    w = weight_fc_all[start_idx:start_idx+size].reshape(shape)
    fc_weights.append(w)
    start_idx += size

start_idx = 0
for shape in fc_shapes:
    size = shape[0]
    b = bias_fc_all[start_idx:start_idx+size]
    fc_biases.append(b)
    start_idx += size

# 初始化PyTorch模型
conv1 = nn.Conv2d(3, 32, 3, padding=1)
conv2 = nn.Conv2d(32, 32, 3, padding=1)
conv3 = nn.Conv2d(32, 64, 3, padding=1)
conv4 = nn.Conv2d(64, 128, 3, padding=1)
pooling = nn.MaxPool2d(2, stride=2)
relu = nn.ReLU()
fc1 = nn.Linear(128 * 5 * 5, 256)
fc2 = nn.Linear(256, 128)
fc3 = nn.Linear(128, 7)

# 加载参数
conv1.weight.data = torch.tensor(conv_weights[0], dtype=torch.float32)
conv1.bias.data = torch.tensor(conv_biases[0], dtype=torch.float32)
conv2.weight.data = torch.tensor(conv_weights[1], dtype=torch.float32)
conv2.bias.data = torch.tensor(conv_biases[1], dtype=torch.float32)
conv3.weight.data = torch.tensor(conv_weights[2], dtype=torch.float32)
conv3.bias.data = torch.tensor(conv_biases[2], dtype=torch.float32)
conv4.weight.data = torch.tensor(conv_weights[3], dtype=torch.float32)
conv4.bias.data = torch.tensor(conv_biases[3], dtype=torch.float32)

fc1.weight.data = torch.tensor(fc_weights[0], dtype=torch.float32)
fc1.bias.data = torch.tensor(fc_biases[0], dtype=torch.float32)
fc2.weight.data = torch.tensor(fc_weights[1], dtype=torch.float32)
fc2.bias.data = torch.tensor(fc_biases[1], dtype=torch.float32)
fc3.weight.data = torch.tensor(fc_weights[2], dtype=torch.float32)
fc3.bias.data = torch.tensor(fc_biases[2], dtype=torch.float32)

# 图像预处理
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# 读取图像
img = Image.open('./data/image.bmp')
img_tensor = transform(img).unsqueeze(0)

# PyTorch前向传播
x = relu(conv1(img_tensor))
x = relu(conv2(x))
x = pooling(x)
x = relu(conv3(x))
x = pooling(x)
x = relu(conv4(x))
feature_conv = pooling(x)
feature_conv = relu(feature_conv)
feature_conv = feature_conv.view(1, -1)
x = relu(fc1(feature_conv))
x = relu(fc2(x))
feature_fc = fc3(x)

# c
c_img = np.loadtxt("../cnn_c/output/image.txt")
rows, cols = c_img.shape
c_img = c_img.reshape(3, rows // 3, cols).astype(np.float32)
p_img = img_tensor.squeeze().numpy()
image_error = np.mean((abs(p_img - c_img)))
print('Error C Image:  {:.10f}'.format(image_error))

c_conv = np.loadtxt("../cnn_c/output/output_conv.txt")
c_conv = c_conv.reshape(128, 5, 5)
conv_c_mean_error = np.mean(
    abs(feature_conv.squeeze().detach().numpy().reshape(128, 5, 5) - c_conv))
print('Error C Conv: {:.10f}'.format(conv_c_mean_error))

c_fc = np.loadtxt("../cnn_c/output/output_fc.txt")
fc_c_mean_error = np.mean(abs(feature_fc.squeeze().detach().numpy() - c_fc))
print('Error Fc: {:.10f}'.format(fc_c_mean_error))

print('finish...')
