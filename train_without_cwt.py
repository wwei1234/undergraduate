from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_without_cwt import mydataset 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
from densenet_without_cwt import DenseNet
import matplotlib.pyplot as plt
import numpy as np

mydata = mydataset('D:/桌面/GD/data/DL_lithology/well_d', 'D:/桌面/GD/data/DL_lithology/inf.xlsx','D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25, 1620)        
# 假设 dataset 是你的数据集实例，包含了所有的样本
dataset = mydata  # 这里填写你的数据集实例
# 定义分割比例
train_ratio = 0.8  # 训练集比例
test_ratio = 0.2   # 测试集比例
# 计算分割的样本数量
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(1)
# 使用 random_split 函数进行分割
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size= 4, shuffle = True,  drop_last=True)
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True,  drop_last=True)
# 
if __name__ == "__main__":
# 创建model，默认是torch.FloatTensor
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet()                                                    #.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    epochs = 2
    train_loss_rate = []
    test_loss_rate = []
    for epoch in range(epochs):
        running_loss_train = 0.0
        running_loss_test = 0.0
        model.train()
        for i, (input, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # 前向过程(model + loss)开启 autocast
            with autocast():
                output = model(input)
                train_loss = nn.CrossEntropyLoss()(output, target.long().squeeze())
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss_train += train_loss.item()
        epoch_loss_train = running_loss_train / len(train_loader)
        train_loss_rate.append(epoch_loss_train)
        ##################################################
        #测试集损失函数
        model.eval() 
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(test_loader)):
                outputs = model(inputs)
                test_loss = nn.CrossEntropyLoss()(outputs, targets.long().squeeze())
                running_loss_test += test_loss.item()    
        epoch_loss_test = running_loss_test/len(test_loader)
        test_loss_rate.append(epoch_loss_test)
        print('train_loss:', epoch_loss_train)
        print('test_loss:', epoch_loss_test)  
    np.save('train_loss_1D_20', train_loss_rate) 
    np.save('test_loss_1D_20', test_loss_rate) 
    ######################################################
    # 绘制训练损失函数曲线
    epochs = range(1, epochs + 1)
    # 绘制训练集和测试集损失函数图表
    plt.plot(epochs, train_loss_rate, 'b', label='Train Loss')
    plt.plot(epochs, test_loss_rate, 'r', label='Test Loss')
    # 添加标题、轴标签和图例
    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #设置保存模型文件的目录路径
    save_dir = 'D:/桌面/GD/data/DL_lithology/models/'  # 假设模型文件存放在名为 models 的子目录下
    # 如果目录不存在，则创建目录
    os.makedirs(save_dir, exist_ok=True)
    # 模型文件的名称
    model_name = 'model121_2_4_1d'
    # 设置保存模型文件的完整路径
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')
    plt.show()
    
    