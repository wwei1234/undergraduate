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
import re


def read_log(well_path):
# 创建一个空列表来存储数据
    filtered_data = []
    # 打开文件
    with open(well_path, 'r', encoding = 'utf') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行末的换行符并按空格分割行数据
            data = line.strip().split()
            # 如果数据行的第一列是数字（排除空行）qq
            if data and data[0].isdigit():
                # 如果是“结论”列，我们需要检查是否需要替换汉字为数字
                filtered_data.append(data[1:3] + [data[-1]])
    # 汉字数字对应的字典
    conversion_dict = {
        '水层': '2',
        '差气层': '1',
        '干层': '0',
        '特低渗气层': '1',
        '气水同层': '2',
        '含气水层': '2',
        '气层':'1'
    }
    # 替换汉字为数字
    for item in filtered_data:
        item[-1] = conversion_dict.get(item[-1], item[-1])
    return filtered_data

def read_inf(inf_path):
    data_dict = {}
    # 打开文件
    with open(inf_path, 'r') as file:
        # 逐行读取文件内容
        for idx, line in enumerate(file):
            # 跳过第一行
            if idx == 0:
                continue
            # 去除行末的换行符并按空格分割行数据
            data = line.strip().split()
            # 如果数据行的长度大于等于3（排除空行或不完整行）
            if len(data) >= 3:
                # 提取前三列数据
                columns = data[:3]
                # 去除冒号
                columns[0] = columns[0].replace(':', '')
                # 将第一个名称作为键，后面两个数据作为值添加到字典中
                data_dict[columns[0]] = (columns[1], columns[2])
    return data_dict

def read_segy_as_data(segyfile, inline_row, xline_row, inline, xline, xline_start):
    with segyio.open(segyfile,"r+",iline=inline_row, xline=xline_row) as sgydata:
        head_data = sgydata.iline[inline][xline-xline_start]         
    return head_data #得到井数据相同位置的地震道数据



class mydataset_hn(Dataset):
    def __init__(self, pathname, inf_path, segyfile, inline_row, xline_row, xline_start, x, y, grid_size, frequency):
        self.pathname = pathname                              #井资料文件夹                                               
        self.inf_path = inf_path
        self.segyfile = segyfile                              #地震资料路径 
        self.inline_row = inline_row
        self.xline_row = xline_row
        self.xline_start = xline_start
        self.x_start = x
        self.y_start = y
        self.grid_size = grid_size
        self.frequency = frequency
        #得到储存有井文件路径的列表
        names = os.listdir(self.pathname)                   
        c = []
        for name in names:
            path = self.pathname + '/' + name
            c.append(path)       
                             
        label = torch.empty([0, 1])
        data = torch.empty([0, self.frequency])
        #生成整口井的时频域数据
        for well_path in c:
            #找出井的inline、xline，并找到对应的地震数据
            match = re.search(r'DF\d+-\d+-[A-Za-z\d]+', well_path)
            well_name = match.group(0)
            well_dict = read_inf(self.inf_path)
            inline = int((int(float(well_dict[well_name][0])) - self.x_start)/self.grid_size) + 2880
            xline = int((int(self.y_start - float(well_dict[well_name][1])))/self.grid_size) + 2870
            data_segy = read_segy_as_data(self.segyfile, self.inline_row, self.xline_row, inline, xline, self.xline_start)

            #seismic depth
            initial_depth = 0 # 地震数据起始深度
            interval = 1 #采样间隔
            depth_segy = (torch.arange(len(data_segy)) + initial_depth)  * interval
            
            #提取井信息
            well_data = read_log(well_path)
            label_temp = torch.empty([0, 1])
            data_temp = torch.empty([0, self.frequency])
            #制作地震切片
            for i in range(len(data_segy)): 
                if i >= self.frequency/2 and i <= len(data_segy) - self.frequency/2 :
                    for sublist in well_data:
                        if float(sublist[0]) <= depth_segy[i] and depth_segy[i] <= float(sublist[1]):
                            label_temp = torch.concat([label_temp, torch.tensor(int(sublist[2])).unsqueeze(0).unsqueeze(0)], dim = 0)
                            features = torch.tensor(data_segy[int(i - self.frequency/2 ): int(i + self.frequency/2)])
                            data_temp = torch.concat([data_temp, features.unsqueeze(0)], dim = 0)              
            data = torch.concat([data, data_temp], dim = 0)    
            label = torch.concat([label, label_temp], dim = 0)
        self.data = data.unsqueeze(1)    
        self.label = label
                 
    def __getitem__(self, index):
        data = self.data[index]        
        label = self.label[index]   
        return data, label
        
    def __len__(self):    
        return len(self.data)


# mydata_hn = mydataset_hn('D:/桌面/2024王威本科毕设/HN/井', 'D:/桌面/2024王威本科毕设/HN/井点坐标.txt','D:/桌面/2024王威本科毕设/HN/CNOOC_DF11_3D_OBN_QPSDM_STK_RAW_D_FEB2023.segy', 189, 193, 2870, 148525.5, 2068537.5, 12.5, 64)        
# mydataloder_hn = DataLoader(mydata_hn, batch_size = 1, shuffle = False)     

# print(len(mydataloder_hn))  
 
        
# save_dir = "D:\桌面\GD\data\HN\dataset_1d"  # 修改为你想要保存的目录路径

# for i, (inputs, targets) in enumerate(mydataloder_hn):
#     # 构建文件名
#     input_filename = os.path.join(save_dir, f'input_batch_{i}.pt')
#     target_filename = os.path.join(save_dir, f'target_batch_{i}.pt')
    
#     # 保存张量数据到磁盘上
#     torch.save(inputs, input_filename)
#     torch.save(targets, target_filename)    
    


