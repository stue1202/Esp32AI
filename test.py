# 导入所需库
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 生成模拟数据
num_samples = 1000
x_values = torch.randn(num_samples, 1) * 5 # 从高斯分布采样x 
true_w = 2.0  # 设置真实的w
true_b = 1.5  # 设置真实的b
y_values = true_w * x_values + true_b + torch.randn(num_samples, 1) # y = wx + b +噪声

# 准备数据
train_data = TensorDataset(x_values, y_values)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出维度都是1
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegressionModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        
        # 前向传播
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
# 评估模型
with torch.no_grad():
    y_pred = model(x_values)
    print(f'True w: {true_w.item()}, Learned w: {model.linear.weight.item():.4f}') 
    print(f'True b: {true_b.item()}, Learned b: {model.linear.bias.item():.4f}')