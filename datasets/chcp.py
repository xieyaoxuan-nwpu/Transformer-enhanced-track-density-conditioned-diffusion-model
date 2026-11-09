import h5py
import numpy as np
from torch.utils.data import Dataset
'''
自定义数据集：训练集和验证集
'''

class TrainDataset(Dataset):#训练集是30通道的数据，但是30通道打乱，所以是单通道数据
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file#指出数据集是哪一个h5文件

        # 预先把(subject, slice_idx, channel_idx)映射展开，方便随机采样
        # 数据结构: hr (30, 30, H, W), lr (30, 30, h, w), fibre_density (30, 1, H, W)
        # 30个z切片，每个切片30个通道
        self.sample_indices = []
        with h5py.File(self.h5_file, 'r') as f:
            for subject in f.keys():
                num_slices = f[subject]['hr'].shape[0]  # 30 z切片
                num_channels = f[subject]['hr'].shape[1]  # 30 通道
                for slice_idx in range(num_slices):
                    for ch_idx in range(num_channels):
                        self.sample_indices.append((subject, slice_idx, ch_idx))

    def __getitem__(self, idx):
        subject, slice_idx, ch_idx = self.sample_indices[idx]
        with h5py.File(self.h5_file, 'r') as f:
            # hr: (30, 30, 140, 140) -> 取单通道: (1, 140, 140)
            # lr: (30, 30, 70, 70) -> 取单通道: (1, 70, 70)
            # fibre_density: (30, 1, 140, 140) -> (1, 140, 140)
            lr = f[subject]['lr'][slice_idx, ch_idx:ch_idx+1, :, :]  # (1, 70, 70)
            hr = f[subject]['hr'][slice_idx, ch_idx:ch_idx+1, :, :]  # (1, 140, 140)
            fibre_density = f[subject]['fibre_density'][slice_idx]  # (1, 140, 140)
            return lr, hr, fibre_density

    def __len__(self):
        return len(self.sample_indices)


class EvalDataset(Dataset):#测试集是30通道数据，没有打乱，所以仍是30通道数据
    def __init__(self, h5_file,number):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.number = number

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:#标准化：Z=(x-均值)/标准差


            lr = f[self.number]['lr'][idx]#[..., None]
            hr = f[self.number]['hr'][idx]#[..., None]
            fibre_density=f[self.number]['fibre_density'][idx]
            return lr,  hr, fibre_density

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f[self.number]['hr'])
