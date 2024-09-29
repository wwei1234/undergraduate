import torch
from densenet_without_cwt import DenseNet, _DenseBlock, _DenseLayer, _Transition
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_without_cwt import mydataset 
import numpy as np

mydata = mydataset('D:/桌面/GD/data/DL_lithology/pre/JS33-5-3', 'D:/桌面/GD/data/DL_lithology/inf.xlsx','D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25, 1620)        
mydataloder = DataLoader(mydata, batch_size=1, shuffle = False)     
# 加载模型
model_path = 'D:/桌面/GD/data/DL_lithology/models/model121_20_4_1d.pth'
model = torch.load(model_path)
# 将模型设置为评估模式
model.eval()
correct = 0
total = 0
predicted_s = torch.empty([0])
target_s = torch.empty([0, 1])

# 在不计算梯度的情况下进行预测
with torch.no_grad():
    for i, (inputs, targets) in enumerate(tqdm(mydataloder)):
        # 将输入数据传递给模型进行预测
        outputs = model(inputs)
        # 获取预测结果（假设是分类任务，取最大概率的类别作为预测结果）
        _, predicted = torch.max(outputs, 1)
        predicted_s = torch.concat((predicted_s, predicted))
        target_s = torch.concat((target_s, targets))
        # 更新总样本数
        total += targets.size(0)
        # 更新正确预测的样本数
        correct += (predicted == targets).sum().item()
# 计算测试集的平均损失值
# 计算准确率
accuracy = correct / total
print('Accuracy: {:.2f}%'.format(100 * accuracy))
##################################################
array = predicted_s.view(-1, 1).numpy()
np.save('predicted.npy', array)
cmap = plt.cm.colors.ListedColormap(['white', 'orange'])
# 绘制图片
plt.imshow(array, cmap=cmap, aspect='auto', extent=[0, 1,  len(target_s)*4 + 515,515])
plt.yticks(np.arange(515,515 + (len(target_s)+1)*4, 100))
plt.xticks([0, 0.5, 1], [0, 0.5, 1])
# 设置标题
plt.title('Predicted')
plt.ylabel('depth/m', labelpad=-2)
# 显示图片
plt.show()


