import torch
import torch.nn as nn

#AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes):  # 传入所需分类的类别数，也就是最后一个全连接层所需要输出的channel
        super(AlexNet, self).__init__()
        # 用torch.nn.Sequential方法将网络打包成一个模块，精简代码，不需要每一步都写成self.conv1 = ...的格式了
        self.features = nn.Sequential(  # 定义卷积层提取图像特征
            # 计算output的size的计算公式：(input_size-kernel_size+padding)/stride + 1
            nn.Conv2d(3, 48, kernel_size=11, padding=2, stride=4),  # input(3, 244, 244) output(48, 55, 55)
            nn.ReLU(inplace=True),  # 直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),  # input(48, 55, 55) output(48, 27, 27)
            nn.Conv2d(48, 128, kernel_size=5, padding=2, stride=1),  # input(48, 27, 27) output(128, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),  # input(128, 27, 27) output(128, 13, 13)
            nn.Conv2d(128, 192, kernel_size=3, padding=1, stride=1),  # input(128, 13, 13) output(192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1),  # input(192, 13, 13) output(192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1),  # input(192, 13, 13) output(128, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),  # input(128, 13, 13) output(128, 6, 6)
        )

        self.classifier = nn.Sequential(
            # 定义全连接层图像分类
            # dropout随机失活神经元，默认比例0.5，一般加在全连接层防止过拟合 提升模型泛化能力。卷积层一般很少加，因为卷积参数少，不易过拟合
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # (batch_size, c , H, W)即从通道c开始对x进行展平
        x = self.classifier(x)
        return x

#VGG
class VGG(nn.Module):
    def __init__(self, num_classes):  # 传入所需分类的类别数，也就是最后一个全连接层所需要输出的channel
        super(VGG, self).__init__()

        def __init__(self, num_classes=1000):
            super(VGG, self).__init__()
            layers = []
            in_dim = 3
            out_dim = 64
            # 循环构造卷积层，一共有13个卷积层
            for i in range(13):
                layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
                in_dim = out_dim
                # 在第2、4、7、10、13个卷积层后增加池化层
                if i == 1 or i == 3 or i == 6 or i == 9 or i == 12:
                    layers += [nn.MaxPool2d(2, 2)]
                    # 第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i != 9:
                    out_dim *= 2
            self.features = nn.Sequential(*layers)
            # VGGNet的3个全连接层，中间有ReLU与Dropout层
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True), nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        # 这里是将特征图的维度从[1, 512, 7, 7]变到[1, 512*7*7]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x