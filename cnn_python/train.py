# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入进度条库

from model import LCNN
from dataset import RafdbDataset

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Rafdb Training')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./data')
opts = parser.parse_args()

# 创建results文件夹
if not os.path.exists('./results'):
    os.makedirs('./results')

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = LCNN(opts).to(device)

# 创建数据加载器
train_dataset = RafdbDataset(opts.data_dir, os.path.join(
    opts.data_dir, 'labels.txt'), mode='train')
test_dataset = RafdbDataset(opts.data_dir, os.path.join(
    opts.data_dir, 'labels.txt'), mode='test')

train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=opts.batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)} 测试集大小: {len(test_dataset)}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opts.lr)

# 训练循环
best_acc = 0.0
for epoch in range(opts.epochs):
    # 训练阶段
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 添加训练进度条
    train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{opts.epochs}] Train',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # 更新进度条描述
        avg_loss = running_loss / \
            (len(train_bar) * opts.batch_size) * inputs.size(0)
        train_acc = 100 * correct_train / total_train
        train_bar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{train_acc:.2f}%')

    # 计算平均训练损失
    train_loss = running_loss / len(train_loader)

    # 测试阶段
    net.eval()
    correct_test = 0
    total_test = 0

    # 添加测试进度条
    test_bar = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{opts.epochs}] Test',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # 更新测试进度条
            test_acc = 100 * correct_test / total_test
            test_bar.set_postfix(acc=f'{test_acc:.2f}%')

    # 计算测试准确率
    test_acc = 100 * correct_test / total_test

    print(
        f'Epoch [{epoch+1}/{opts.epochs}], Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'net_state_dict': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch+1
        }, './results/best.ckpt')
        print(
            f'Saved best model with acc: {best_acc:.2f}% at epoch {epoch+1}!')

print('Finished Training')
print(f'Best Test Accuracy: {best_acc:.2f}%')
