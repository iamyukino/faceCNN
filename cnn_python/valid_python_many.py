
import os
import argparse
import numpy as np
from PIL import Image

from valid_python_single import read_bin_as_floats, func_conv2d, func_max_pooling, func_relu, func_fc, transform


if __name__ == '__main__':

    weight_conv = np.array(read_bin_as_floats('./params/weight_conv.bin')).reshape(8, 3, 3, 3)
    bias_conv   = np.array(read_bin_as_floats('./params/bias_conv.bin'))
    weight_fc   = np.array(read_bin_as_floats('./params/weight_fc.bin')).reshape(7, 200)
    bias_fc     = np.array(read_bin_as_floats('./params/bias_fc.bin'))

    # batch inference
    img_dir = './data/images'
    label_path = './data/labels.txt'

    N = 100
    preds = []
    targets = []
    with open(label_path, 'r', encoding = 'utf - 8') as f:
        lines = f.readlines()
        for line in lines[-N:]:
            img_name, label = line.split(' ')
            img = Image.open(os.path.join(img_dir, img_name))
            img_tensor = transform(img).unsqueeze(0) 
            output_conv = func_conv2d(img_tensor.squeeze().numpy().astype(np.float32), weight_conv, bias_conv, stride=1, padding=1)
            output_conv = func_relu(func_max_pooling(output_conv, 8, 8))
            output_fc   = func_fc(output_conv.reshape(1, -1), weight_fc, bias_fc)
            preds.append(np.argmax(output_fc, 1).item())
            targets.append(int(label)-1)

    acc = 0
    for v1, v2 in zip(preds, targets):
        acc += 1 if v1 == v2 else +0
    print(f"准确率(Accuracy): {acc}")
