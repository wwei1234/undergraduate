import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import segyio
import numpy as np
import os
import pandas as pd
import pywt
import matplotlib
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
matplotlib.use('TkAgg')
import re
import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def read_inf(inf_path):
    df1 = pd.read_excel(inf_path,sheet_name = 'Sheet1', header = 0)
    well_dict = df1.set_index('well_name').to_dict(orient='index')
    return well_dict

def read_segy_as_data(segyfile, inline_row, xline_row, inline, xline, xline_start):
    with segyio.open(segyfile,"r+",iline=inline_row, xline=xline_row) as sgydata:
        head_data = sgydata.iline[inline][xline-xline_start]         
    return head_data                                      #得到井数据相同位置的地震道数据

def cwts(cwts_scale, sampling_rate, data):
    #数据
    wavename = "mexh"  # 小波函数
    totalscal = cwts_scale + 1  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam/np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    cwtmatr = pywt.cwt(data, scales, wavename, 1.0/sampling_rate)[0]  # 连续小波变换模块
    feature_maps = cwtmatr
    return  feature_maps                                   #进行连续小波变换得到时频域地震信息

def depth_time(td_path, well_name):
    well_name = well_name + '.csv'
    file_path = os.path.join(td_path, well_name)
    # 使用pandas读取csv文件
    df = pd.read_csv(file_path)
    # 打印数据框的内容
    td = df.iloc[:, :2].values
    depth = td[:, 0]
    times = td[:, 1]
    return depth, times

def read_results(results_path, well_name):
    well_name = well_name + '.txt'
    file_path = os.path.join(results_path, well_name)
    data = pd.read_csv(file_path, sep='\t', encoding = 'gbk')
    selected_data = data[['顶深', '底深', '名称']]
    # 将选定的数据转换为列表
    result_list = selected_data.values.tolist()
    # 汉字到数字的映射字典
    mapping_dict = {
        '油层': 1,
        '油水同层': 1,
        '含油水层': 1,
        '水层': 2,
        '水淹层': 2,
        '弱淹水层': 2,
        '气水同层': 2,
        '干层': 0,
        '差油层': 1,
        '中淹水层': 2,
        '强淹水层': 2,
        '可疑气层': 0,
        '较强水淹层': 2,
        '泥岩裂缝储层': 1
    }
    # 干层；含油层（油层，含油水层，油水同层）；含水层
    # 替换列表中的汉字为数字
    for entry in result_list:
        entry[2] = mapping_dict.get(entry[2], entry[2])
    return result_list

class mydataset_sb(Dataset):
    def __init__(self, td_path, cwts_scale, sampling_rate, inf_path, segy_path, inline_row, xline_row, xline_start, result_path, frequency):
        self.td_path = td_path                              #井资料文件夹                                               
        self.sampling_rate = sampling_rate
        self.inf_path = inf_path
        self.segy_path = segy_path                              #地震资料路径 
        self.inline_row = inline_row
        self.xline_row = xline_row
        self.xline_start = xline_start
        self.cwts_scale = cwts_scale
        self.result_path = result_path
        self.frequency = frequency
        
        data = torch.empty([0, self.frequency, self.frequency])
        label = torch.empty([0, 1])
        
        well_dict = read_inf(self.inf_path)
        for well_name, values in well_dict.items():
            inline = values['inline']
            xline = values['xline']   
            #读取地震数据
            data_segy = read_segy_as_data(self.segy_path, self.inline_row, self.xline_row, inline, xline, self.xline_start)                 
            feature_maps = cwts(self.cwts_scale, self.sampling_rate, data_segy)
            feature_maps = torch.tensor(feature_maps)
            #找到具有时深关系的地震数据段
            [depth_segy, time] = depth_time(self.td_path, well_name)
            feature_maps = feature_maps[:, int(time[0]) : int(time[-1])]
            #根据时深关系计算地震数据的深度信息
            depth_segy = torch.tensor(depth_segy)
            interp_func = interp1d(x=range(len(depth_segy)), y=depth_segy, kind='linear')
            new_indices = torch.linspace(0, len(depth_segy) - 1, feature_maps.shape[1])
            depth_segy = interp_func(new_indices)
            
            #读入解释数据
            result_list = read_results(self.result_path, well_name)
            #制作地震切片和标签
            label_temp = torch.empty([0, 1])
            data_temp = torch.empty([0, self.frequency, self.frequency])
            for i in range(feature_maps.shape[1]): 
                if i >= self.frequency/2 and i <= feature_maps.shape[1] - self.frequency/2 :
                    for sublist in result_list:
                        if float(sublist[0]) <= depth_segy[i] and depth_segy[i] <= float(sublist[1]):
                            label_temp = torch.concat([label_temp, torch.tensor(int(sublist[2])).unsqueeze(0).unsqueeze(0)], dim = 0)
                            features = feature_maps[-1 - self.frequency : -1, int(i - self.frequency/2) : int(i + self.frequency/2)]
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

# mydata_sb = mydataset_sb('D:/桌面/SB/时深关系统计', 128, 1000, 'D:/桌面/SB/inf.xlsx','D:/桌面/SB/shengbei_pstm_cg.sgy', 9, 21,890, 'D:/桌面/SB/油水层统计', 64)        
# mydataloder = DataLoader(mydata_sb, batch_size= 1, shuffle = False)   
# print(len(mydataloder))  
     
# save_dir = "D:\桌面\GD\data\SB\dataset"  # 修改为你想要保存的目录路径

# for i, (inputs, targets) in enumerate(mydataloder):
#     # 构建文件名
#     input_filename = os.path.join(save_dir, f'input_batch_{i}.pt')
#     target_filename = os.path.join(save_dir, f'target_batch_{i}.pt')
    
#     # 保存张量数据到磁盘上
#     torch.save(inputs, input_filename)
#     torch.save(targets, target_filename)    
   








