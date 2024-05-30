import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST  #训练集六万张，测试集一万张，每张图片28*28个像素点，每个像素的灰度值在0-255之间，每张图片有一个标记即其数字值
import matplotlib.pyplot as plt
'''
实现子module: Residual Block
'''
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias = False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias = False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
'''
实现主module: ResNet34
ResNet34包含多个layer, 每个layer又包含多个residual block
用子module实现residual block, 用_make_layer函数实现layer
'''
class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 3, groups=1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride = 2)
        self.layer3 = self._make_layer(256, 512, 6, stride = 2)
        self.layer4 = self._make_layer(512, 512, 3, stride = 2)
        
        # 分类用的全连接
        self.fc = nn.Linear(2048, num_classes)
        
    def _make_layer(self, inchannel, outchannel, block_num, stride = 1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias = False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()]) #define the type of the data 
    data_set = MNIST("",is_train,transform = to_tensor,download=True)
    return DataLoader(data_set, batch_size = 15, shuffle = True) # 15 pictures in 1 batch & the data is random


def main():
    train_data = get_data_loader(is_train = True)
    test_data = get_data_loader(is_train = False)
    net = ResNet() #初始化神经网络
    
    optimizer = t.optim.Adam(net.parameters(), lr = 0.001) # set the parameters of the optimizer(learning rate equals to 0.001)
    for epoch in range(2):  # 2 times loops
        for (x, y) in train_data:
            net.zero_grad()  #初始化
            output = net.forward(x.view(-1, 3, 28, 28)) #正向传播
            loss = t.nn.functional.nll_loss(output, y) #计算差值
            loss.backward() #反向误差传播
            optimizer.step() #优化参数
        
        
    for (n,(x,_)) in enumerate(test_data):  #随机抽取3张图片进行测试 /enumerate 用于一个可以遍历的数据对象
        if n > 3:
            break
        predict = t.argmax(net.forward(x[0].view(28, 28))) 
        plt.figure(n)
        plt.imshow(x[0].view(28,28))
        plt.title("prediction:" + str(int(predict)))
    plt.show()  
    
    
main()