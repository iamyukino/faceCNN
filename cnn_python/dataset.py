# dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class RafdbDataset(Dataset):
    def __init__(self, data_dir, label_path, mode='train', transform=None):
        self.data_dir = data_dir

        # 保存传入的transform对象
        self.transform = transform if transform else self.get_default_transform()

        # 读取标签文件
        self.labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(' ')
                # 根据模式过滤图像
                if mode == 'train' and img_name.startswith('train_'):
                    self.labels[img_name] = int(label) - 1  # 标签转换为0-based索引
                elif mode == 'test' and img_name.startswith('test_'):
                    self.labels[img_name] = int(label) - 1

        # 获取符合模式的图像列表
        self.image_list = list(self.labels.keys())

        # 情绪标签映射
        self.emotion_map = ['surprise', 'fear', 'disgust',
                            'happiness', 'sadness', 'anger', 'neutral']

    def get_default_transform(self):
        """默认转换（如果没有外部传入transform）"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_name]
        return image, label
