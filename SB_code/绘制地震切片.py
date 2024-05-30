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
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
#连续小波变换
############################################################################################################
# def read_segy_as_data(segyfile, inline_row, xline_row, inline, xline, xline_start):
#     with segyio.open(segyfile,"r+",iline=inline_row, xline=xline_row) as sgydata:
#         head_data = sgydata.iline[inline][xline-xline_start]         
#     return head_data 

# def cwts(cwts_scale, sampling_rate, data):
#     #数据
#     wavename = "mexh"  # 小波函数
#     totalscal = cwts_scale + 1  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
#     fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
#     cparam = 2 * fc * totalscal  # 常数c
#     scales = cparam/np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
#     #scales = np.arange(0,40,30/128)
#     cwtmatr = pywt.cwt(data, scales, wavename, 1.0/sampling_rate)[0]  # 连续小波变换模块
#     feature_maps = cwtmatr
#     return  feature_maps  

# data = read_segy_as_data('D:/桌面/GD/data/DL_lithology/CX_ZJ_ori.sgy', 1, 25, 3209, 1752, 1620)
# feature = cwts(128, 50, data)
# feature = feature[-65 : -1, :]
# feature = feature.transpose()
# feature = np.flip(feature, axis=1)
# time_range = np.linspace(0, 800, 401)  # 生成从 0 到 800 ms 的时间范围
# 
# plt.subplot(1,2,1)
# plt.plot(data, time_range)   
# plt.ylabel('Time (ms)', fontsize = 20)
# plt.title('一维时间域地震信息', fontsize = 20)
# plt.tick_params(axis='both', which='major', labelsize=16)  # 设置刻度值字体大小为 12
# plt.gca().invert_yaxis()  # 反转纵轴
# custom_ticks = [ -20000, -10000, 0, 10000, 20000]
# custom_tick_labels = [str(abs(tick)) if tick != 0 else '0' for tick in custom_ticks]
# # 设置横坐标刻度及标签
# plt.xticks(custom_ticks, custom_tick_labels)

# plt.subplot(1,2,2)
# plt.imshow(feature, extent=[0, 64, 800, 0], aspect='auto', cmap='seismic')
# plt.title('连续小波变换后', fontsize = 20)
# plt.ylabel('Time (ms)', fontsize = 20)
# plt.tick_params(axis='both', which='major', labelsize=16)  # 设置刻度值字体大小为 12
# plt.show()

# 地震切片
##########################################################################################
save_folder = 'D:\桌面\GD\data\SB\images_2'
os.makedirs(save_folder, exist_ok=True)
for i in range(0, 1000):
    # 格式化文件路径，替换数字
    input_path = f'D:\桌面\GD\data\SB\dataset/input_batch_{i}.pt'
    target_path = f'D:\桌面\GD\data\SB\dataset/target_batch_{i}.pt'
    # 读取文件
    input_data = torch.load(input_path).squeeze()
    target_data = torch.load(target_path).squeeze()
    target_data = int(target_data)
    input= input_data.numpy()
    input = input.transpose()
    input= np.flip(input, axis=1)
    mapping_dict = {
        1: '油层',
        2: '水层',
        0: '干层'
    }
    # mapping_dict = {
    #     1: '砂岩',
    #     0: '泥岩'
    # }
    # mapping_dict = {
    #     1: '气层',
    #     2: '水层',
    #     0: '干层'
    # }
    plt.imshow(input, aspect='equal', cmap='seismic')
    # plt.title(f'地震切片{i}', fontsize = 20)
    plt.xlabel(r'Frequency(Hz)' + '\t' + f'标签：{mapping_dict[target_data]}', fontsize=20)
    plt.ylabel('Time(ms)', fontsize = 20)
    plt.savefig(f'D:\桌面\GD\data\SB\images_2\slice_{i}_{mapping_dict[target_data]}.png')
    # 清空当前图像
    plt.clf()
    print(f'切片{i}已保存')
    
    
    
    
    
    
    
    
    
    
    
    
    
#损失函数
##############################################################################################3
# train_loss = np.load('D:\GD/train_loss_model121_70_4.npy')
# train_loss_c = np.load('D:\GD/train_loss_model121_70_4_continue.npy')
# train_loss = np.append(train_loss, train_loss_c)
# plt.plot(train_loss, label = 'train loss')
# plt.legend(['train loss'], fontsize = 20)
# plt.xlabel('epochs (次)', fontsize = 20)
# plt.ylabel('loss', fontsize = 20)
# plt.title('损失函数曲线', fontsize = 20)
# plt.show()


