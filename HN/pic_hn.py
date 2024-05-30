import torch
from densenet_hn import DenseNet, _DenseBlock, _DenseLayer, _Transition
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_hn import mydataset_hn 
import numpy as np
import matplotlib.colors as mcolors
import segyio
import pywt
from matplotlib.colors import ListedColormap
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
mydata_hn = mydataset_hn('D:/桌面/2024王威本科毕设/HN/DF1-1-1C', 128, 500, 'D:/桌面/2024王威本科毕设/HN/井点坐标.txt','D:/桌面/2024王威本科毕设/HN/CNOOC_DF11_3D_OBN_QPSDM_STK_RAW_D_FEB2023.segy', 189, 193, 2870, 148525.5, 2068537.5, 12.5, 64)        
mydataloder = DataLoader(mydata_hn, batch_size = 1, shuffle = False)     
# 加载模型
model_path = 'D:\桌面\GD\data\HN\models/model121_30_2.pth'
model = torch.load(model_path)
# 设置模型为评估模式
model.eval()
# 假设 test_loader 是测试集的 DataLoader，model 是已经训练好的模型
correct = 0
total = 0
predicted_s = torch.empty([0])
target_s = torch.empty([0, 1])
# 在不计算梯度的情况下进行预测
with torch.no_grad():
    for i, (inputs, targets) in enumerate(tqdm(mydataloder)):
        # 将输入数据传递给模型进行预测
        outputs = model(inputs)
        # 获取预测结果（假设是分类任务，取最大概率的类别作为预测结果）
        _, predicted = torch.max(outputs, 1)
        predicted_s = torch.concat((predicted_s, predicted))
        target_s = torch.concat((target_s, targets))
        # 更新总样本数
        total += targets.size(0)
        # 更新正确预测的样本数
        correct += (predicted == targets).sum().item()
# 计算准确率
accuracy = correct / total
print('Accuracy: {:.2f}%'.format(100 * accuracy))
print(total)
#################################################################################
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
    return head_data 

def cwts(cwts_scale, sampling_rate, data):
    #数据
    wavename = "mexh"  # 小波函数
    totalscal = cwts_scale + 1  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam/np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    #scales = np.arange(0,40,30/128)
    cwtmatr = pywt.cwt(data, scales, wavename, 1.0/sampling_rate)[0]  # 连续小波变换模块
    feature_maps = cwtmatr
    return  feature_maps  
x_start = 157097.5 
y_start = 2055952.7
ts = 602
td = 2663
grid_size = 12.5
well_dict = read_inf('D:/桌面/2024王威本科毕设/HN/井点坐标.txt')
well_name = 'DF1-1-1C'
inline = int((int(float(well_dict[well_name][0])) - x_start)/grid_size) + 2880
xline = int((int(y_start - float(well_dict[well_name][1])))/grid_size) + 2870
data = read_segy_as_data('D:/桌面/2024王威本科毕设/HN/CNOOC_DF11_3D_OBN_QPSDM_STK_RAW_D_FEB2023.segy', 189, 193, inline, xline, 2870)
feature = cwts(128, 50, data)
feature = feature[-65 : -1, :]
feature = feature.transpose()
feature = np.flip(feature, axis=1)
feature = feature[ts : td, :]
time_range = np.linspace(ts , td, td-ts)  # 生成从 0 到 800 ms 的时间范围

data = data[ts:td]
plt.subplot(1,4,1)
plt.plot(data, time_range)   
# plt.ylabel('Time (ms)', fontsize = 20)
plt.ylabel('Depth (m)', fontsize = 20)
plt.title('原始地震信号', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置刻度值字体大小为 12
plt.gca().invert_yaxis()  # 反转纵轴
# custom_ticks = [ -20000,0, 20000]
# custom_tick_labels = [str(abs(tick)) if tick != 0 else '0' for tick in custom_ticks]
# # 设置横坐标刻度及标签
# plt.xticks(custom_ticks, custom_tick_labels)

plt.subplot(1,4,2)
plt.imshow(feature, extent=[0, 64, td,ts], aspect='auto', cmap='seismic')
plt.title('连续小波变换后', fontsize = 20)
plt.xlabel('Frequency(Hz)', fontsize = 20)
# plt.ylabel('Depth(m)', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置刻度值字体大小为 12
# ###################################################################################################
# ###################################################################################################
# 创建图形
array = predicted_s.view(-1, 1).numpy()
cmap = plt.cm.colors.ListedColormap(['gray', 'orange','blue'])
# 绘制图片
plt.subplot(1, 4, 3)
plt.imshow(array, cmap=cmap, aspect='auto', extent=[0, 64, td,ts])
# 设置标题
plt.title('预测结果', fontsize = 20)
plt.xticks([])
plt.yticks(fontsize=14)
# plt.ylabel('Depth(m)', fontsize = 14)
##################################################################################################
# array = np.load('D:\GD\predict_resnet_hn.npy')
# plt.subplot(1, 3,2)
# plt.imshow(array, cmap=cmap, aspect='auto', extent=[0, 64, td,ts])
# # 设置标题
# plt.title('ResNet预测结果', fontsize = 20)
# plt.xticks([])
# plt.yticks(fontsize=14)
##################################################################################################
cmap = plt.cm.colors.ListedColormap(['gray', 'orange','blue'])
array = target_s.view(-1, 1).numpy()
# 绘制图片
plt.subplot(1, 4, 4)
plt.imshow(array, cmap=cmap, aspect='auto', extent=[0, 64,  td,ts])
# 设置标题
plt.title('Target', fontsize = 20)
plt.xticks([])
plt.yticks(fontsize=14)
# plt.ylabel('depth/m', labelpad=-2)
# 显示图片
fig = plt.gcf()  # 获取当前图形
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]
cMap = ListedColormap(['blue', 'gray', 'orange'])
cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cMap), cax=cbar_ax, ticks=[0.25, 0.5, 0.75])
cb.ax.set_yticklabels(['水层', '干层', '气层'])
cb.ax.tick_params(labelsize=16)  # 设置颜色条标签的字体大小

plt.subplots_adjust(right=0.9)  # 调整右侧边距以留出颜色条空间
plt.show()









