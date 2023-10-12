import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

"""
# GPU时可用
# 定义构造数据加载器的函数
def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集函数
def LoadCIFAR10(download=False):
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform)
    return train_dataset, test_dataset
"""

"""
AlexNet数据库导入：图像+标签
"""

class My_Dataset(Dataset):
    def __init__(self, data_root, transform=None, data='train', methods='Alex'):
        super(My_Dataset, self).__init__()
        self.transform = transform
        self.filename = data_root

        if data == 'train':
            self.filename += '/train'
            self.image_name, self.label_image = self.operate_file()
        else:
            self.filename += '/test'

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        # 由路径打开图片
        image = Image.open(self.image_name[idx])
        # 下采样： 因为图片大小不同，需要下采样为227*227
        trans = transforms.RandomResizedCrop(227)
        image = trans(image)
        # 获取标签值
        label = self.label_image[idx]
        # 是否需要处理
        if self.transform:
            image = self.transform(image)
        # 转为tensor对象
        label = torch.from_numpy(np.array(label))
        return image, label

    def operate_file(self):
        # 获取所有的文件夹路径 '../data/train'下的文件夹
        dir_list = os.listdir(self.filename)
        # 拼凑出图片完整路径 '../data/train' + '/' + 'xxx.jpg'
        full_path = [self.filename + '/' + name for name in dir_list]
        # 获取里面的图片名字
        name_list = []
        for i, v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v + '/' + j for j in temp]
            name_list.extend(temp_list)
        # 由于一个文件夹的所有标签都是同一个值，而字符值必须转为数字值，因此我们使用数字0-4代替标签值
        # 将标签每个复制200个
        label_list = []
        temp_list = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 用数字代表不同类别
        for j in range(5):
            for i in range(200):
                label_list.append(temp_list[j])
        return name_list, label_list