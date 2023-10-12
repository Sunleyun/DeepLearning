import torch
from CNN import Model
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 预处理
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 导入要测试的图像（自己找的，不在数据集中），放在源文件目录下
img = Image.open("./ftest2.jpg")
plt.imshow(img)  # 通过PIL或者numpy导入的图片格式一般都是[H, W, C](高度、宽度、通道)
img = transform(img)  # 通过transform转换成[C, H, W]
# 对数据增加一个维数为1的新维度，因为tensor的参数是[batch, channel, height, width]
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = Model(num_classes=5)
# 加载训练好的权重文件
model.load_state_dict(torch.load("Alexnet.pth"))

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # 预测分类
    # output = torch.squeeze(model(img))
    # 将输出压缩，即压缩掉 batch 这个维度
    # 如果这样去做，print(predict)输出的就是一个一维数组的tensorflow，所以后面的dim=1应该改成dim=0
    # 最后输出的predict[0][predict_cla].item()直接写成predict[predict_cla].item()就行
    output = model(img)
    predict = torch.softmax(output, dim=1)
    print(predict)
    predict_cla = torch.argmax(predict).numpy()  # argmax返回最大值的索引
    print(predict_cla)
print(class_indict[str(predict_cla)],
      predict[0][predict_cla].item())  # 通过输出的predict我们可以看出predict是一个二维数组形式的tensor，所以需要写成predict[0][predict_cla]
plt.show()