import  torch  as  t
from  torch  import  nn
from  torch.nn  import  functional  as  F
from  torch.utils.data  import  DataLoader
from  torchvision  import  transforms
from  torchvision.datasets  import  MNIST
import  matplotlib.pyplot  as  plt

class  ResidualBlock(nn.Module):
    def  __init__(self,  inchannel,  outchannel,  stride=1,  shortcut=None):
        super(ResidualBlock,  self).__init__()
        self.left  =  nn.Sequential(
        nn.Conv2d(inchannel,  outchannel,  3,  stride,  1,  bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),
        nn.Conv2d(outchannel,  outchannel,  3,  1,  1,  bias=False),
        nn.BatchNorm2d(outchannel))
        self.right  =  shortcut
        
    def  forward(self,  x):
        out  =  self.left(x)
        residual  =  x  if  self.right  is  None  else  self.right(x)
        out  +=  residual
        return  F.relu(out)

class  ResNet(nn.Module):
    def  __init__(self,  num_classes=10):
        super(ResNet,  self).__init__()
        self.pre  =  nn.Sequential(
            nn.Conv2d(3,  64,  3,  2,  3,  groups=1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1  =  self._make_layer(64,  128,  3)
        self.layer2  =  self._make_layer(128,  256,  4,  stride=2)
        self.layer3  =  self._make_layer(256,  512,  6,  stride=2)
        self.layer4  =  self._make_layer(512,  512,  3,  stride=2)

        self.fc  =  nn.Linear(2048,  num_classes)

    def  _make_layer(self,  inchannel,  outchannel,  block_num,  stride=1):
        shortcut  =  nn.Sequential(
            nn.Conv2d(inchannel,  outchannel,  1,  stride,  bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers  =  []
        layers.append(ResidualBlock(inchannel,  outchannel,  stride,  shortcut))

        for  i  in  range(1,  block_num):
            layers.append(ResidualBlock(outchannel,  outchannel))
        return  nn.Sequential(*layers)

    def  forward(self,  x):
        x  =  self.pre(x)

        x  =  self.layer1(x)
        x  =  self.layer2(x)
        x  =  self.layer3(x)
        x  =  self.layer4(x)

        x   =   x.view(x.size(0),  -1)
        return   self.fc(x)

def  get_data_loader(is_train):
    to_tensor  =  transforms.Compose([transforms.ToTensor()])
    data_set  =  MNIST("",is_train,transform=to_tensor,download=True)
    return   DataLoader(data_set,   batch_size=15,   shuffle=True)

def  main():
    train_data  =  get_data_loader(is_train=True)
    test_data  =  get_data_loader(is_train=False)
    net  =  ResNet()

    optimizer  =  t.optim.Adam(net.parameters(),   lr=0.001)
    criterion  =  nn.CrossEntropyLoss()

    for   epoch   in   range(10):
        for   iteration,   (x,   y)   in   enumerate(train_data):
            net.zero_grad()
            outputs  =  net(x.view(-1,   3,   28,   28))
            loss  =  criterion(outputs,   y)
            loss.backward()
            optimizer.step()

        print(f"Epoch  {epoch  +  1}/10,   Loss:  {loss.item()}")

    for   iteration,   (x,   y)   in   enumerate(test_data):
        outputs  =  net(x[0].view(-1,   3,   28,   28))
        predicted  =  t.argmax(outputs,   dim=1)
        print(f"Predicted:  {predicted.item()}")

    plt.show()

main()
