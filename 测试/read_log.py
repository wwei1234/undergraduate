import numpy as np
import os
import matplotlib.pyplot as plt

def read_resample_mud_log(file ,interval):
    with open(file, 'r') as log:   ##一口井
        content = log.read()
    extracted_data = []
    for line in content.split('\n'):
        if line.strip() and line.strip()[0].isdigit():
            parts = line.split()
            well_section_start = parts[1]
            well_section_end = parts[2]
            well_result = parts[-1] 
            extracted_data.append((well_section_start, well_section_end, well_result))
    well_name = os.path.basename(file)[0:-4]  ###井名
    depth_start = extracted_data[0][0]
    depth_end = extracted_data[-1][1]
    depth = np.arange(int(float(depth_start)), int(float(depth_end))+1, interval)      ###深度
    mud = np.zeros(np.shape(depth))                                 ###录井
    for start, end, interpretation in extracted_data:
        i = 0
        for deep in depth:                              ###1420
            if float(start) <= deep <= float(end):      
                if interpretation == '干层':
                    mud[i] = 0
                elif interpretation == '差气层':
                    mud[i] = 1
                elif interpretation == '特低渗气层':
                    mud[i] = 2
                elif interpretation == '气水同层':
                    mud[i] = 3
                elif interpretation == '气层':
                    mud[i] = 4
                elif interpretation == '含气水层':
                    mud[i] = 5
                elif interpretation == '水层':
                    mud[i] = 6
            i = i+1  
    return well_name, depth, mud

def mud_log_plot(mud,depth,colorbar):
    data = mud.reshape(len(mud),1)
    plt.figure(figsize=(2,10))
    plt.imshow(data,interpolation='nearest', aspect='auto',extent=(0,1,depth[-1],depth[0]))  
    plt.xticks([])
    if colorbar:
        plt.colorbar()
    plt.show()
    
