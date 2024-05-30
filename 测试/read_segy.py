import numpy as np
import segyio

def read_segy_as_data(filepath, inline_row, xline_row):
    
    with segyio.open(filepath,"r+",iline=inline_row, xline=xline_row) as sgydata:
        data = segyio.tools.cube(sgydata)
    return data


def output_segy(filepath, inline_row, xline_row, data):
    ## input data's shape must be the same with output data
    with segyio.open(filepath,"r+",iline=inline_row, xline=xline_row) as sgydata:
        a = 0
        for i in sgydata.xlines:
            sgydata.xline[i] = data[a]
            a = a + 1
            
def load_np_seismic(path, dim):
    seismic=np.fromfile(path,dtype=np.float32)
    seismic=seismic.reshape(dim)
    seismic = np.expand_dims(seismic, axis=0)        # add batch_size=1 for 3D data --> [1, Xline, Inline, Time]
    seismic = np.expand_dims(seismic, axis=4)        # add channel=1 for 3D data    --> [1, Xline, Inline, Time, 1]
    seismic=seismic.transpose(0, 3, 1, 2, 4)         # [batch_size=1, depth=Time, height=Xline, width=Inline, channel=1]
    return seismic

def extract_trace_from_segy(filepath, inline_row, xline_row, inline, xline):
    num_inline = inline
    num_xline = xline
    with segyio.open(filepath,"r+",iline=inline_row, xline=xline_row) as sgydata:
        # data = segyio.tools.cube(sgydata)
        sgydata.mmap()
        # print(sgydata.ilines)
        # print(sgydata.xlines)
        trace_seis_data = sgydata.iline[num_inline][num_xline-sgydata.xlines[0]]
    return trace_seis_data

data = read_segy_as_data('C:/Users/wwod/Desktop/新建文件夹/test.sgy',189,193)
print(data)