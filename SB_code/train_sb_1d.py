from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_1 import mydataset_sb 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
from densenet_sb_1d import DenseNet
import matplotlib.pyplot as plt
import numpy as np

mydata_sb_1d = mydataset_sb('D:/桌面/SB/时深关系统计', 'D:/桌面/SB/inf.xlsx','D:/桌面/SB/shengbei_pstm_cg.sgy', 9, 21,890, 'D:/桌面/SB/油水层统计', 64)        
# dataset = mydata_sb  
# # 定义分割比例
# train_ratio = 0.8  # 训练集比例
# test_ratio = 0.2   # 测试集比例
# # 计算分割的样本数量
# train_size = int(train_ratio * len(dataset))
# test_size = len(dataset) - train_size
# torch.manual_seed(42)
# # 使用 random_split 函数进行分割
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# # 创建 DataLoader 加载数据
# train_loader = DataLoader(train_dataset, batch_size= 4, shuffle = False)
# test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)
train_loader = DataLoader(mydata_sb_1d, batch_size = 4, shuffle = True)
if __name__ == "__main__":
# 创建model，默认是torch.FloatTensor
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet()   
    # model_path = 'D:/桌面/GD/data/SB/models\model_sb_20_4.pth'
    # model = torch.load(model_path)                                   #.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    epochs = 150
    train_loss_rate = []
    test_loss_rate = []
    for epoch in range(epochs):
        print(epoch)
        running_loss_train = 0.0
        model.train()
        for i, (input, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # 前向过程(model + loss)开启 autocast
            with autocast():
                output = model(input)
                train_loss = nn.CrossEntropyLoss()(output, target.long().view(-1))
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss_train += train_loss.item()
        epoch_loss_train = running_loss_train / len(train_loader)
        train_loss_rate.append(epoch_loss_train)
        ##################################################
        # #测试集损失函数
        # model.eval()   
        # with torch.no_grad():
        #     for i, (inputs, targets) in enumerate(tqdm(test_loader)):
        #         outputs = model(inputs)
        #         test_loss = nn.CrossEntropyLoss()(outputs, targets.long().squeeze())
        #         running_loss_test += test_loss.item()    
        # epoch_loss_test = running_loss_test/len(test_loader)
        # test_loss_rate.append(epoch_loss_test)
        print('train_loss:', epoch_loss_train)
        # print('test_loss:', epoch_loss_test) 
    np.save('train_loss_sb_150', train_loss_rate) 
    # np.save('test_loss_sb', test_loss_rate)
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
    save_dir = 'D:/桌面/GD/data/SB/models'  # 假设模型文件存放在名为 models 的子目录下
    # 如果目录不存在，则创建目录
    os.makedirs(save_dir, exist_ok=True)
    # 模型文件的名称
    model_name = 'model_sb_1d'
    # 设置保存模型文件的完整路径
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')
    plt.show()



