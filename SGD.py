from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_DL import mydataset 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
from densenet1 import DenseNet
import matplotlib.pyplot as plt
import numpy as np

mydata = mydataset('D:/桌面/GD/data/DL_lithology/well_d', 128, 500, 'D:/桌面/GD/data/DL_lithology/inf.xlsx','D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25, 1620, 64)            
dataset = mydata  
train_loader = DataLoader(mydata, batch_size= 4, shuffle = True, drop_last=True)
if __name__ == "__main__":
# 创建model，默认是torch.FloatTensor
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet()                                                    #.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    epochs = 20
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
        print('train_loss:', epoch_loss_train)
    np.save('train_loss_dl_00001_20', train_loss_rate)  
    ######################################################
    # 绘制训练损失函数曲线
    epochs = range(1, epochs + 1)
    # 绘制训练集和测试集损失函数图表
    plt.plot(epochs, train_loss_rate, 'b', label='Train Loss')
    # plt.plot(epochs, test_loss_rate, 'r', label='Test Loss')
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
    model_name = 'model121_DL_00001_20'
    # 设置保存模型文件的完整路径 
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')
    plt.show()



