
import os
import torch
from train.model import LCNN
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Rafdb')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--resume_net', type=str, default='./results/best.ckpt')
opts = parser.parse_args()

net = LCNN(opts)   
ckpt = torch.load(opts.resume_net, map_location=lambda storage, loc: storage)
net.load_state_dict(ckpt['net_state_dict'])

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])    
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize           
])
img = Image.open('./data/image.bmp')
img = transform(img).unsqueeze(0)   
logits = net(img)

if os.path.isdir('./params') is False:
    os.mkdir('./params')

conv_out = net.conv(img)
pool_out = net.pool(conv_out)
relu_out = net.relu(pool_out)
relu_out = relu_out.view(1, -1)
fc_out   = net.fc(relu_out)

# here
weight_conv = net.conv.weight.detach().numpy().astype(np.float32)
bias_conv = net.conv.bias.detach().numpy().astype(np.float32)
weight_conv.reshape(-1).tofile('./params/weight_conv.bin')
bias_conv.tofile('./params/bias_conv.bin')

weight_fc = net.fc.weight.detach().numpy().astype(np.float32)
bias_fc = net.fc.bias.detach().numpy().astype(np.float32)
weight_fc.reshape(-1).tofile('./params/weight_fc.bin')
bias_fc.tofile('./params/bias_fc.bin')

error = np.mean(abs(logits.detach().numpy() - fc_out.detach().numpy()))
print('Logits: {}'.format(logits.data))
print('Error Logits: {:.10f}'.format(error))

_, predict = torch.max(logits, 1)
label2emotion = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
print('Prediction: {}'.format(label2emotion[predict.item()]))




