import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 创建输入数据
x = np.linspace(-5, 5, 400)

# 计算激活函数的输出
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)

# 绘制Sigmoid
plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid_y, label='Sigmoid', color='blue', linewidth=2)
plt.title('Sigmoid', fontsize=16)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 绘制Tanh
plt.figure(figsize=(8, 6))
plt.plot(x, tanh_y, label='Tanh', color='red', linewidth=2)
plt.title('Tanh', fontsize=16)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 绘制ReLU
plt.figure(figsize=(8, 6))
plt.plot(x, relu_y, label='ReLU', color='green', linewidth=2)
plt.title('ReLU', fontsize=16)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 绘制Leaky ReLU
plt.figure(figsize=(8, 6))
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='purple', linewidth=2)
plt.title('Leaky ReLU', fontsize=16)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
