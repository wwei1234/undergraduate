import torch
from torch.utils.data import Dataset
import numpy as np
import segyio
import numpy as np
import os
import pandas as pd
import pywt
import matplotlib
import torch.nn.functional as F
from torch.utils.data import DataLoader
matplotlib.use('TkAgg')
def read_log(well_path):
        with open(well_path, 'r') as log:   
            content = log.read()
        extracted_data = []
        for line in content.split('\n'):
            if line.strip() and line.strip()[0].isdigit():
                parts = line.split()
                well_result = parts[-1]
                well_time = parts[-2] 
                if int(float(well_result)) >= 0 and int(float(well_time)) >=0 :
                    extracted_data.append((well_result, well_time))
        well_name = os.path.basename(well_path)  ###井名
        mud = np.array(extracted_data)[:,0]
        well_time = np.array(extracted_data)[:,-1] ###录井
        return well_name, mud, well_time
    
def read_inf(inf_path, well_name):
    df1 = pd.read_excel(inf_path,sheet_name = 'Sheet1', header = 0)
    index1 = df1['name'].tolist().index(well_name)
    inline = df1['inline'][index1]                                               
    xline = df1['xline'][index1]
    return inline, xline

def read_segy_as_data(segyfile, inline_row, xline_row, inline, xline, xline_start):
    with segyio.open(segyfile,"r+",iline=inline_row, xline=xline_row) as sgydata:
        head_data = sgydata.iline[inline][xline-xline_start]         
    return head_data                                     #得到井数据相同位置的地震道数据

class mydataset(Dataset):
    def __init__(self, pathname, inf_path, segyfile, inline_row, xline_row, xline_start):
        self.pathname = pathname                              #井资料文件夹                                               
        self.inf_path = inf_path
        self.segyfile = segyfile                              #地震资料路径 
        self.inline_row = inline_row
        self.xline_row = xline_row
        self.xline_start = xline_start
        
        
        #得到储存有井文件路径的列表
        names = os.listdir(self.pathname)                   
        c = []
        for name in names:
            path = self.pathname + '/' + name
            c.append(path)                        
        
        #初始化一个空张量用来储存data和label    
        data = torch.empty([0, 64])               
        label = torch.empty([0 , 1])
        
        #生成整口井的时频域数据
        for well_path in c:
            [well_name, mud, well_time] = read_log(well_path)
            [inline, xline] = read_inf(self.inf_path, well_name)
            data_segy = read_segy_as_data(self.segyfile, self.inline_row, self.xline_row, inline, xline, self.xline_start)
            
            #well time
            well_time = well_time.astype(float)
            well_time = torch.tensor(well_time)  
            #seismic time
            initial_depth = 0 # 地震数据起始深度
            interval = 2
            time_segy = (torch.arange(len(data_segy)) + initial_depth)  * interval
            
            #找到地震数据和井数据对应的一部分
            min_val = max(well_time[0], time_segy[0] + (64/2) * interval)
            max_val = min(well_time[-1], time_segy[-1] - (64/2) * interval)
                         
            if min_val < max_val :
                start_index = torch.argmin(torch.abs(time_segy - min_val))
                end_index = torch.argmin(torch.abs(time_segy - max_val))
                #将对应的时域数据截取出来
                data_segy = data_segy[int(start_index - 64/2) : int(end_index + 64/2)] 
                number_of_images = data_segy.shape[0] - 64
                
                #对井数据进行插值,将label与data匹配
                start_index = torch.argmin(torch.abs(well_time - min_val))
                end_index = torch.argmin(torch.abs(well_time - max_val))
                mud = mud.astype(float)
                mud = torch.tensor(mud[start_index : end_index])
                mud = F.interpolate(mud.unsqueeze(0).unsqueeze(0), size = number_of_images, mode='linear').squeeze().int() 
                mud = mud.unsqueeze(1)   
                
                #初始化每口井的data张量              
                image = torch.empty([0,64]) 
                
                for i in range(number_of_images):
                    image_later = torch.tensor(data_segy[i : i + 64])  #截取大小为（x，x）的张量 
                    image_later  = torch.unsqueeze(image_later, 0)        #这里把截取的图像变为(1,x)
                    image = torch.concat([image,image_later], 0)              
                
                    
                data = torch.concat([data, image], dim = 0)    
                label = torch.concat([label, mud], dim = 0)
            else : 
                continue
                
        self.data = data.unsqueeze(1) #添加一个维度用作channel
        self.label = label
    
            
    def __getitem__(self, index):
        data = self.data[index]        
        label = self.label[index]   
        return data, label
        
    def __len__(self):    
        return len(self.data)


# mydata = mydataset('D:/桌面/GD/data/DL_lithology/well_d', 'D:/桌面/GD/data/DL_lithology/inf.xlsx','D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25,1620)        
# mydataloder = DataLoader(mydata, batch_size = 1, shuffle = False)     
# print(len(mydataloder))  
      
# save_dir = "D:/桌面/GD/data/DL_lithology/dataset_1D"  # 修改为你想要保存的目录路径

# for i, (inputs, targets) in enumerate(mydataloder):
#     # 构建文件名
#     input_filename = os.path.join(save_dir, f'input_batch_{i}.pt')
#     target_filename = os.path.join(save_dir, f'target_batch_{i}.pt')
    
#     # 保存张量数据到磁盘上
#     torch.save(inputs, input_filename)
#     torch.save(targets, target_filename) 

