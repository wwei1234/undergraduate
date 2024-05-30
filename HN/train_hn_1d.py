from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_hn_1d import mydataset_hn 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
from densenet_hn_1d import DenseNet
import matplotlib.pyplot as plt
import numpy as np

mydata_hn_1d = mydataset_hn('D:/桌面/2024王威本科毕设/HN/well_hn', 'D:/桌面/2024王威本科毕设/HN/井点坐标.txt','D:/桌面/2024王威本科毕设/HN/CNOOC_DF11_3D_OBN_QPSDM_STK_RAW_D_FEB2023.segy', 189, 193, 2870, 148525.5, 2068537.5, 12.5, 64)          
train_loader = DataLoader(mydata_hn_1d, batch_size = 4, shuffle = True, drop_last= True)
# test_loader = DataLoader(test_dataset, batch_size = 6, shuffle = True)
if __name__ == "__main__":
# 创建model，默认是torch.FloatTensor
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'D:\桌面\GD\data\HN\models/model121_hn_1d_3.pth'
    model = torch.load(model_path)                                                   #.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    epochs = 10
    train_loss_rate = []
    test_loss_rate = []
    for epoch in range(epochs):
        print(epoch)
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
    np.save('train_loss_hn_final', train_loss_rate)  
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
    save_dir = 'D:\桌面\GD\data\HN\models/'  # 假设模型文件存放在名为 models 的子目录下
    # 如果目录不存在，则创建目录
    os.makedirs(save_dir, exist_ok=True)
    # 模型文件的名称
    model_name = 'model121_hn_1d_4'
    # 设置保存模型文件的完整路径
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')
    plt.show()




