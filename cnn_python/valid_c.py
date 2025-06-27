
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


img = Image.open('./data/image.bmp')
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
img_tensor = transform(img).unsqueeze(0)

weight_conv = np.array(read_bin_as_floats(
    './params/weight_conv.bin')).reshape(8, 3, 3, 3)
bias_conv = np.array(read_bin_as_floats('./params/bias_conv.bin'))
weight_fc = np.array(read_bin_as_floats(
    './params/weight_fc.bin')).reshape(7, 200)
bias_fc = np.array(read_bin_as_floats('./params/bias_fc.bin'))

conv = nn.Conv2d(3, 8, 3, 1, 1)
fc = nn.Linear(200, 7)
pooling = nn.MaxPool2d(8, stride=8)
relu = nn.ReLU()

conv.weight.data = torch.tensor(weight_conv, dtype=torch.float32)
conv.bias.data = torch.tensor(bias_conv,   dtype=torch.float32)
fc.weight.data = torch.tensor(weight_fc,   dtype=torch.float32)
fc.bias.data = torch.tensor(bias_fc,     dtype=torch.float32)

feature_conv = relu(pooling(conv(img_tensor)))
feature_fc = fc(feature_conv.view(1, -1))

# c
c_img = np.loadtxt("../cnn_c/output/image.txt")
rows, cols = c_img.shape
c_img = c_img.reshape(3, rows // 3, cols).astype(np.float32)
p_img = img_tensor.squeeze().numpy()
image_error = np.mean((abs(p_img - c_img)))
print('Error C Image:  {:.10f}'.format(image_error))

temp_img = 255 * (0.5 * c_img + 0.5)
temp_img = Image.fromarray(temp_img.transpose(1, 2, 0).astype(np.uint8))
temp_img.save('temp_img.jpg')

c_conv = np.loadtxt("../cnn_c/output/output_conv.txt")
channel, rows, cols = feature_conv.squeeze().shape
c_conv = c_conv.reshape(channel, rows, cols)
conv_c_mean_error = np.mean(
    abs(feature_conv.squeeze().detach().numpy() - c_conv))
print('Error C Conv: {:.10f}'.format(conv_c_mean_error))

c_fc = np.loadtxt("../cnn_c/output/output_fc.txt")
channel, rows = feature_fc.shape
c_fc = c_fc.reshape(channel, rows)
fc_c_mean_error = np.mean(abs(feature_fc.squeeze().detach().numpy() - c_fc))
print('Error Fc: {:.10f}'.format(fc_c_mean_error))

print('finish...')
