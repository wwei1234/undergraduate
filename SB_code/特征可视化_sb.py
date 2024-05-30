import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dataset_sb_images import mydataset_sb 
import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image
from densenet_sb import DenseNet, _DenseBlock, _DenseLayer, _Transition
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
model_path = 'D:\桌面\GD\data\SB\models/model121_80_4.pth'
model = torch.load(model_path)
model.eval()
traget_layers_1 = [model.features.pool0]
input_2000 = torch.load('D:\桌面\GD\data\SB\dataset/input_batch_700.pt')
########################################################################################################
input_2000_1 = input_2000.squeeze()
input_2000_1= input_2000_1.numpy()
input_2000_1 = input_2000_1.transpose()
input_2000_1= np.flip(input_2000_1, axis=1)
plt.subplot(1,2,1)
plt.imshow(input_2000_1, aspect='equal', cmap='seismic')
plt.title('输入层输入', fontsize = 20)
#########################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
feature_map = None
def hook_fn(module, input, output):
    global feature_map
    feature_map = output.detach().cpu()
######################################################################################################33
#选择需要可视化的层
layer_to_visualize = model.features.pool0
# 在指定层注册钩子
hook = layer_to_visualize.register_forward_hook(hook_fn)
output = model(input_2000)
feature = feature_map[0, 0, :, :]
feature = feature.squeeze().squeeze()
feature = feature.numpy()
feature = feature.transpose()
feature= np.flip(feature, axis=1)
plt.subplot(1,2,2)
plt.imshow(feature, aspect='equal', cmap='seismic')
plt.title('输入层输出', fontsize = 20)
########################################################################################################
# layer_to_visualize = model.features.denseblock1.denselayer1.conv2
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,2,2)
# plt.imshow(feature, aspect='equal', cmap='seismic')
# plt.title('密集层1输出', fontsize = 20)
########################################################################################################
# layer_to_visualize = model.features.denseblock1.denselayer6.conv2
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,2,2)
# plt.imshow(feature, aspect='equal', cmap='seismic')
# plt.title('密集块1输出', fontsize = 20)
########################################################################################################
# layer_to_visualize = model.features.denseblock2.denselayer12.conv2
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,5,3)
# plt.imshow(feature, aspect='equal', cmap='seismic')
# plt.title('密集块2输出', fontsize = 20)
########################################################################################################
# layer_to_visualize = model.features.denseblock3.denselayer24.conv2
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,5,4)
# plt.imshow(feature, aspect='equal', cmap='seismic', extent = [0, 4, 4, 0])
# plt.gca().set_xticks([0, 1, 2, 3, 4])
# plt.gca().set_yticks([0, 1, 2, 3, 4])
# plt.title('密集块3输出', fontsize = 20)
#################################################################################################33333#
# layer_to_visualize = model.features.transition1.pool
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,2,2)
# plt.imshow(feature, aspect='equal', cmap='seismic')
# plt.title('过渡层1输出', fontsize = 20)
##########################################################################################################3
# layer_to_visualize = model.features.denseblock4.denselayer16.conv2
# # 在指定层注册钩子
# hook = layer_to_visualize.register_forward_hook(hook_fn)
# output = model(input_2000)
# feature = feature_map[0, 0, :, :]
# feature = feature.squeeze().squeeze()
# feature = feature.numpy()
# feature = feature.transpose()
# feature= np.flip(feature, axis=1)
# plt.subplot(1,5,5)
# plt.imshow(feature, aspect='equal', cmap='seismic', extent = [0, 2, 2, 0])
# plt.gca().set_xticks([0, 1, 2])
# plt.gca().set_yticks([0, 1, 2])
# plt.title('密集块4输出', fontsize = 20)

plt.show()





