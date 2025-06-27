
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from model import LCNN
from dataset import RafdbDataset

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Rafdb Training')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=128)  # 使用大批大小
parser.add_argument('--epochs', type=int, default=30)  # 30轮训练
parser.add_argument('--lr', type=float, default=0.01)  # 较高学习率
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--data_dir', type=str, default='./data')
opts = parser.parse_args()

# 创建results文件夹
if not os.path.exists('./results'):
    os.makedirs('./results')

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = LCNN(opts).to(device)

# 计算并打印模型大小
total_params = sum(p.numel() for p in net.parameters())
model_size_bytes = total_params * 4  # 每个参数4字节
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"模型参数数量: {total_params} (约{model_size_mb:.2f}MB)")

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 创建数据加载器
train_dataset = RafdbDataset(opts.data_dir, os.path.join(opts.data_dir, 'labels.txt'),
                             mode='train', transform=train_transform)
test_dataset = RafdbDataset(opts.data_dir, os.path.join(opts.data_dir, 'labels.txt'),
                            mode='test', transform=test_transform)

train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_dataset)} 测试集大小: {len(test_dataset)}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=opts.lr,
                        weight_decay=opts.weight_decay)

# 使用余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=opts.epochs, eta_min=1e-4)

# 训练循环
best_acc = 0.0
for epoch in range(opts.epochs):
    # 训练阶段
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_bar = tqdm(
        train_loader, desc=f'Epoch [{epoch+1}/{opts.epochs}] Train')

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
        train_acc = 100 * correct_train / total_train
        train_bar.set_postfix(
            loss=f'{loss.item():.4f}', acc=f'{train_acc:.2f}%')

    # 计算平均训练损失
    train_loss = running_loss / len(train_loader)

    # 更新学习率
    scheduler.step()

    # 测试阶段
    net.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    test_bar = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{opts.epochs}] Test')

    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # 更新测试进度条
            test_acc = 100 * correct_test / total_test
            test_bar.set_postfix(acc=f'{test_acc:.2f}%')

    # 计算测试准确率
    test_acc = 100 * correct_test / total_test
    test_loss /= len(test_loader)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch+1}/{opts.epochs}], LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # 保存最佳模型（仅保存状态字典）
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(net.state_dict(), './results/best.ckpt')
        print(
            f'Saved best model with acc: {test_acc:.2f}% at epoch {epoch+1}!')

print('Finished Training')
print(f'Best Test Accuracy: {best_acc:.2f}%')
