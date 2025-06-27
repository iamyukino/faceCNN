
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import struct


# 读取二进制参数文件
def read_bin_as_floats(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    floats = np.frombuffer(data, dtype=np.float32)
    return floats


# 完成卷积操作
def func_conv2d(input, kernel, bias, stride=1, padding=0):
    # 输入形状: [C, H, W]
    _, in_c, in_h, in_w = input.shape
    out_c, _, k, _ = kernel.shape

    # 添加padding
    if padding > 0:
        pad_x = np.zeros((in_c, in_h+2*padding, in_w +
                         2*padding), dtype=np.float32)
        for c in range(in_c):
            pad_x[c] = np.pad(input[0, c], padding, mode='constant')
        x_padded = pad_x
    else:
        x_padded = input[0]

    out_h = (in_h + 2*padding - k) // stride + 1
    out_w = (in_w + 2*padding - k) // stride + 1
    output = np.zeros((1, out_c, out_h, out_w), dtype=np.float32)

    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride
                w_start = ow * stride
                h_end = h_start + k
                w_end = w_start + k

                # 提取输入区域
                region = x_padded[:, h_start:h_end, w_start:w_end]

                # 计算卷积结果
                conv_sum = np.sum(region * kernel[oc])
                output[0, oc, oh, ow] = conv_sum + bias[oc]
    return output


# 完成池化操作
def func_max_pooling(input, kernel_size, stride):
    bs, c, h, w = input.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    output = np.zeros((bs, c, out_h, out_w), dtype=np.float32)

    for b in range(bs):
        for ch in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    region = input[b, ch, h_start:h_end, w_start:w_end]
                    output[b, ch, oh, ow] = np.max(region)
    return output


# ReLU函数
def func_relu(x):
    return np.maximum(0, x)


# 全连接函数
def func_fc(x, weights, biases):
    return np.dot(x, weights.T) + biases


# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
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

    # 加载融合后的参数
    conv_weights = []
    conv_biases = []
    weight_conv_data = read_bin_as_floats('./params/weight_conv.bin')
    bias_conv_data = read_bin_as_floats('./params/bias_conv.bin')

    start_idx = 0
    for shape in conv_shapes:
        size = np.prod(shape)
        w = weight_conv_data[start_idx:start_idx+size].reshape(shape)
        conv_weights.append(w)
        start_idx += size

    start_idx = 0
    for shape in conv_shapes:
        size = shape[0]
        b = bias_conv_data[start_idx:start_idx+size]
        conv_biases.append(b)
        start_idx += size

    fc_weights = []
    fc_biases = []
    weight_fc_data = read_bin_as_floats('./params/weight_fc.bin')
    bias_fc_data = read_bin_as_floats('./params/bias_fc.bin')

    start_idx = 0
    for shape in fc_shapes:
        size = np.prod(shape)
        w = weight_fc_data[start_idx:start_idx+size].reshape(shape)
        fc_weights.append(w)
        start_idx += size

    start_idx = 0
    for shape in fc_shapes:
        size = shape[0]
        b = bias_fc_data[start_idx:start_idx+size]
        fc_biases.append(b)
        start_idx += size

    # 读取并预处理图像
    img = Image.open('./data/image.bmp')
    img_tensor = transform(img).unsqueeze(0)

    # init model
    conv = nn.Conv2d(3, 32, 3, 1, 1)
    conv2 = nn.Conv2d(32, 32, 3, 1, 1)
    conv3 = nn.Conv2d(32, 64, 3, 1, 1)
    conv4 = nn.Conv2d(64, 128, 3, 1, 1)

    fc1 = nn.Linear(128 * 5 * 5, 256)
    fc2 = nn.Linear(256, 128)
    fc3 = nn.Linear(128, 7)

    pooling = nn.MaxPool2d(2, stride=2)
    relu = nn.ReLU()

    # load para
    conv.weight.data = torch.tensor(conv_weights[0], dtype=torch.float32)
    conv.bias.data = torch.tensor(conv_biases[0], dtype=torch.float32)
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

    # torch inference
    x1 = relu(conv(img_tensor))
    x1 = relu(conv2(x1))
    x1 = pooling(x1)
    x1 = relu(conv3(x1))
    x1 = pooling(x1)
    x1 = relu(conv4(x1))
    feature_conv = pooling(x1)
    feature_conv = relu(feature_conv)

    x1 = feature_conv.view(1, -1)
    x1 = relu(fc1(x1))
    x1 = relu(fc2(x1))
    feature_fc = fc3(x1)

    # numpy inference
    x = img_tensor.numpy().copy()

    # 第一卷积块
    x = func_conv2d(x, conv_weights[0], conv_biases[0], stride=1, padding=1)
    x = func_relu(x)
    x = func_conv2d(x, conv_weights[1], conv_biases[1], stride=1, padding=1)
    x = func_relu(x)
    x = func_max_pooling(x, kernel_size=2, stride=2)

    # 第二卷积块
    x = func_conv2d(x, conv_weights[2], conv_biases[2], stride=1, padding=1)
    x = func_relu(x)
    x = func_max_pooling(x, kernel_size=2, stride=2)

    # 第三卷积块
    x = func_conv2d(x, conv_weights[3], conv_biases[3], stride=1, padding=1)
    x = func_relu(x)
    x = func_max_pooling(x, kernel_size=2, stride=2)
    output_conv = func_relu(x)

    # 全连接块
    x = x.reshape(1, -1)
    x = func_fc(x, fc_weights[0], fc_biases[0])
    x = func_relu(x)
    x = func_fc(x, fc_weights[1], fc_biases[1])
    x = func_relu(x)
    output_fc = func_fc(x, fc_weights[2], fc_biases[2])

    # prediction
    predict = np.argmax(output_fc, 1)
    label2emotion = ['surprise', 'fear', 'disgust',
                     'happiness', 'sadness', 'anger', 'neutral']
    print('Logits: {}'.format(output_fc))
    print('Prediction: {}'.format(label2emotion[predict.item()]))

    # error
    conv_mean_error = np.mean(
        abs(feature_conv.squeeze().detach().numpy() - output_conv))
    fc_mean_error = np.mean(abs(feature_fc.detach().numpy() - output_fc))
    print('Error Conv: {:.10f}'.format(conv_mean_error))
    print('Error Fc:   {:.10f}'.format(fc_mean_error))

    print('finish...')
