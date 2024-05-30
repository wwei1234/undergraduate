import torch
from densenet_hn import DenseNet, _DenseBlock, _DenseLayer, _Transition
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset_hn import mydataset_hn 
from sklearn.metrics import precision_score, recall_score, f1_score
mydata_hn = mydataset_hn('D:/桌面/2024王威本科毕设/HN/well_hn', 128, 500, 'D:/桌面/2024王威本科毕设/HN/井点坐标.txt','D:/桌面/2024王威本科毕设/HN/CNOOC_DF11_3D_OBN_QPSDM_STK_RAW_D_FEB2023.segy', 189, 193, 2870, 148525.5, 2068537.5, 12.5, 64)             
dataset = mydata_hn  
# 定义分割比例
train_ratio = 0.8  # 训练集比例
test_ratio = 0.2   # 测试集比例
# 计算分割的样本数量
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(42)
# 使用 random_split 函数进行分割
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size= 10, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

# 加载模型
model_path = 'D:\桌面\GD\data\HN\models/model121_60_resnet.pth'
model = torch.load(model_path)

# 将模型设置为评估模式
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
# 在不计算梯度的情况下进行预测
with torch.no_grad():
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):
        # 将输入数据传递给模型进行预测
        outputs = model(inputs)
        # 获取预测结果（假设是分类任务，取最大概率的类别作为预测结果）
        _, predicted = torch.max(outputs, 1)
        # 更新总样本数
        total += targets.size(0)
        # 更新正确预测的样本数
        correct += (predicted == targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
# 计算准确率
accuracy = correct / total
print('Accuracy: {:.2f}%'.format(100 * accuracy))
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


