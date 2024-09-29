from collections import OrderedDict
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

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))   # 64
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,   
                                           kernel_size=1, stride=1, bias=False))   #(64, 4*32)    bn_size = 4 
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))  #(128)
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))  #(128, 32)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)  
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # concat input and output

class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):  
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)   
            self.add_module("denselayer%d" % (i+1,), layer)     # put per layer to one block


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))     #  =  num_input_feature + num_block * growrate
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes = 2):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):    # (6, 12, 24, 16)
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:   # (4 - 1)
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer    
        self.classifier = nn.Linear(num_features, num_classes)  

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)             # self.feature is the final model
        out = F.avg_pool2d(features, 2, stride=1).view(features.size(0), -1)   # 
        out = self.classifier(out)
        # out = torch.sigmoid(out)
        return out





mydata = mydataset('D:/桌面/GD/data/DL_lithology/well_d', 128, 500, 'D:/桌面/GD/data/DL_lithology/inf.xlsx','D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25, 1620, 64)            
dataset = mydata  
# 定义分割比例 
train_ratio = 0.1  # 训练集比例
# 计算分割的样本数量
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(1)
# 使用 random_split 函数进行分割
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size= 6, shuffle = True)    
    
    
if __name__ == "__main__":
# 创建model，默认是torch.FloatTensor
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet()                                                    #.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    epochs = 300
    train_loss_rate_adam = []
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
        train_loss_rate_adam.append(epoch_loss_train)
        print('train_loss:', epoch_loss_train)
        # print('test_loss:', epoch_loss_test)
    # 绘制训练集和测试集损失函数图表
    model = DenseNet()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
    train_loss_rate_sgd = []
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
        train_loss_rate_sgd.append(epoch_loss_train)
        print('train_loss:', epoch_loss_train)
        # print('test_loss:', epoch_loss_test)
    # 绘制训练损失函数曲线
    epochs = range(1, epochs + 1)
    # plt.title('Training and Test Loss Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    plt.figure(figsize=(12, 6))
    plt.plot(epochs,train_loss_rate_adam, label='Adam', color='b')
    plt.plot(epochs,train_loss_rate_sgd, label='SGD', color='r')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curve Comparison: Adam vs SGD', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()
    
    
    
    
    
    

    
   
        
