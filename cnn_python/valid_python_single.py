
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import struct


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
    return floats


def normalize(tensor):
    tensor = tensor / 255.
    return tensor


# 完成边缘补充操作
def np_pad(array, pad_num):
    # PyTorch的Conv2d(padding=1)表示在高度和宽度方向各填充1
    return np.pad(array,
                  ((0, 0), (pad_num, pad_num), (pad_num, pad_num)),
                  mode='constant')


# 完成卷积操作
def func_conv2d(input, kernel, bias, stride=1, padding=0):
    # 输入形状: [C, H, W]
    C, H, W = input.shape
    out_channels, in_channels, kH, kW = kernel.shape

    # 添加padding
    if padding > 0:
        input = np_pad(input, padding)
        C, H, W = input.shape

    # 计算输出尺寸
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    # 创建输出数组
    output = np.zeros((out_channels, out_H, out_W), dtype=np.float32)

    # 执行卷积
    for oc in range(out_channels):
        for oh in range(out_H):
            for ow in range(out_W):
                h_start = oh * stride
                w_start = ow * stride
                h_end = h_start + kH
                w_end = w_start + kW

                # 提取输入区域
                region = input[:, h_start:h_end, w_start:w_end]

                # 计算卷积结果
                conv_sum = np.sum(region * kernel[oc])
                output[oc, oh, ow] = conv_sum + bias[oc]

    return output


# 完成池化操作
def func_max_pooling(input, win_size, stride, padding=0):
    # 输入形状: [C, H, W]
    C, H, W = input.shape

    # 添加padding（PyTorch默认padding=0）
    if padding > 0:
        input = np_pad(input, padding)
        C, H, W = input.shape

    # 计算输出尺寸
    out_H = (H - win_size) // stride + 1
    out_W = (W - win_size) // stride + 1

    # 创建输出数组
    output = np.zeros((C, out_H, out_W), dtype=np.float32)

    # 执行池化
    for c in range(C):
        for oh in range(out_H):
            for ow in range(out_W):
                h_start = oh * stride
                w_start = ow * stride
                h_end = h_start + win_size
                w_end = w_start + win_size

                # 提取输入区域
                region = input[c, h_start:h_end, w_start:w_end]

                # 计算最大值
                output[c, oh, ow] = np.max(region)

    return output


# 完成激活操作
def func_relu(input):
    return np.maximum(input, 0)


# 完成激活操作
def func_fc(input, weight, bias):
    # 输入形状: [1, features]
    # 权重形状: [out_features, in_features]
    # 偏置形状: [out_features]

    # 矩阵乘法: output = input × weight^T + bias
    output = np.dot(input, weight.T) + bias
    return output.reshape(1, -1)


normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


if __name__ == '__main__':

    img = Image.open('./data/image.bmp')
    img_tensor = transform(img).unsqueeze(0)

    # read para
    weight_conv = np.array(read_bin_as_floats(
        './params/weight_conv.bin')).reshape(8, 3, 3, 3)
    bias_conv = np.array(read_bin_as_floats('./params/bias_conv.bin'))
    weight_fc = np.array(read_bin_as_floats(
        './params/weight_fc.bin')).reshape(7, 200)
    bias_fc = np.array(read_bin_as_floats('./params/bias_fc.bin'))

    # init model
    conv = nn.Conv2d(3, 8, 3, 1, 1)
    fc = nn.Linear(200, 7)
    pooling = nn.MaxPool2d(8, stride=8)
    relu = nn.ReLU()

    # load para
    conv.weight.data = torch.tensor(weight_conv, dtype=torch.float32)
    conv.bias.data = torch.tensor(bias_conv,   dtype=torch.float32)
    fc.weight.data = torch.tensor(weight_fc,   dtype=torch.float32)
    fc.bias.data = torch.tensor(bias_fc,     dtype=torch.float32)

    # torch inference
    feature_conv = relu(pooling(conv(img_tensor)))
    feature_fc = fc(feature_conv.view(1, -1))

    # numpy inference
    output_conv = func_conv2d(img_tensor.squeeze().numpy().astype(
        np.float32), weight_conv, bias_conv, stride=1, padding=1)
    output_conv = func_relu(func_max_pooling(output_conv, 8, 8))
    output_fc = func_fc(output_conv.reshape(1, -1), weight_fc, bias_fc)

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
