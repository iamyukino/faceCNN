import torch.nn as nn


class LCNN(nn.Module):
    def __init__(self, opts):
        super(LCNN, self).__init__()

        # 优化卷积部分
        self.conv_block = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(3, 32, 3, padding=1),  # 减少通道数
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x40 -> 20x20

            # 第2层卷积
            nn.Conv2d(32, 64, 3, padding=1),  # 减少通道数
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x20 -> 10x10

            # 第3层卷积
            nn.Conv2d(64, 128, 3, padding=1),  # 减少通道数
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 10x10 -> 5x5
        )

        # 优化全连接部分
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),  # 大幅减少参数
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(opts.dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(opts.dropout),

            nn.Linear(128, opts.num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x
