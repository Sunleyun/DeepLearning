#train1:一元线性拟合
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 从文件中读取数据
data = np.loadtxt('ex1data1.txt', delimiter=',')
x_data = torch.tensor(data[:, 0].reshape(-1, 1), dtype=torch.float32)
y_data = torch.tensor(data[:, 1].reshape(-1, 1), dtype=torch.float32)

# 定义线性模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

epochs = 1000
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 开始k-折交叉验证
val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
    print(f'Fold {fold + 1}/{k_folds}')

    X_train, y_train = x_data[train_idx], y_data[train_idx]
    X_val, y_val = x_data[val_idx], y_data[val_idx]

    model = LinearRegressionModel()
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

# 结果可视化
predicted = model(x_data).detach().numpy()
plt.figure(figsize=(8, 6))
plt.scatter(x_data.numpy(), y_data.numpy(), color='blue', label='Original data')
plt.plot(x_data.numpy(), predicted, color='red', label='Fitted line')
plt.xlabel('Area')
plt.ylabel('Profit')
plt.legend()
plt.show()

# 预测
area = torch.tensor([[3.1415]], dtype=torch.float32)
predicted_profit = model(area)

print(f"预计在面积为 {area.item()} 的城市开一家餐厅的利润为: {predicted_profit.item()}")