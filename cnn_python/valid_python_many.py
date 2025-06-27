
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

from valid_python_single import read_bin_as_floats, func_conv2d, func_max_pooling, func_relu, func_fc, transform

# 定义网络结构参数
conv_shapes = [
    (32, 3, 3, 3),    # conv1
    (32, 32, 3, 3),   # conv2
    (64, 32, 3, 3),   # conv3
    (128, 64, 3, 3)   # conv4
]
fc_shapes = [
    (256, 128 * 5 * 5),  # fc1
    (128, 256),          # fc2
    (7, 128)             # fc3
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


# 批量推理 (numpy 前向传播)
def numpy_forward(x):
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
    x = x.reshape(x.shape[0], -1)  # 展平
    x = func_fc(x, fc_weights[0], fc_biases[0])
    x = func_relu(x)
    x = func_fc(x, fc_weights[1], fc_biases[1])
    x = func_relu(x)
    output_fc = func_fc(x, fc_weights[2], fc_biases[2])

    return output_fc


if __name__ == '__main__':

    # batch inference
    img_dir = './data/images'
    label_path = './data/labels.txt'

    N = 100
    preds = []
    targets = []
    with open(label_path, 'r', encoding='utf - 8') as f:
        lines = f.readlines()
        for line in tqdm(lines[-N:], desc="Processing images"):
            img_name, label = line.strip().split(' ')
            img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).numpy()  # [1, 3, 40, 40]

            # 前向传播
            output_fc = numpy_forward(img_tensor)

            # 预测结果
            predict = np.argmax(output_fc, axis=1)[0]
            preds.append(predict)
            targets.append(int(label)-1)  # 标签转换为0-based

    acc = 0
    for v1, v2 in zip(preds, targets):
        acc += 1 if v1 == v2 else +0
    print(f"准确率(Accuracy): {acc}")
