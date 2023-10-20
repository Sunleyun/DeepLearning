#train2:多元线性拟合
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

# 从文件中读取数据
data = np.loadtxt('ex1data2.txt', delimiter=',')
x_data = torch.tensor(data[:, :-1], dtype=torch.float32) # 前两列为输入特征
y_data = torch.tensor(data[:, -1].reshape(-1, 1), dtype=torch.float32) # 最后一列为输出

# 定义多元线性回归模型
class MultiLinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

input_dim = x_data.shape[1]
epochs = 30000
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 开始k-折交叉验证
val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
    print(f'Fold {fold + 1}/{k_folds}')

    X_train, y_train = x_data[train_idx], y_data[train_idx]
    X_val, y_val = x_data[val_idx], y_data[val_idx]

    model = MultiLinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
        print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')

print(f'\nAverage Validation Loss: {np.mean(val_losses):.4f}')
print(f'Standard Deviation of Validation Loss: {np.std(val_losses):.4f}')

#结果可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据
ax.scatter(x_data[:, 0].numpy(), x_data[:, 1].numpy(), y_data.numpy(), color='blue', label='Sample Data')

# 为了绘制拟合平面，我们需要创建一个网格
x1 = torch.linspace(min(x_data[:, 0]), max(x_data[:, 0]), 100)
x2 = torch.linspace(min(x_data[:, 1]), max(x_data[:, 1]), 100)
X1, X2 = torch.meshgrid(x1, x2)
Z = model(torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)).reshape(100, 100).detach().numpy()

ax.plot_surface(X1.numpy(), X2.numpy(), Z, alpha=0.5, color='red', label='Fitted Plane')
ax.set_title('3D Scatter Plot with Fitted Plane')
ax.set_xlabel('Area')
ax.set_ylabel('Number')
ax.set_zlabel('Price')
plt.show()

# 使用模型预测
sample = torch.tensor([[2000, 1]], dtype=torch.float32) # 随意选择一个样本，第一个值为面积，第二个值为其他数量
predicted_profit = model(sample)
print(f"面积大小为 {sample[0][0].item()} 和卧室数量为 {sample[0][1].item()} 的房屋成交价格为: {predicted_profit.item()}")

