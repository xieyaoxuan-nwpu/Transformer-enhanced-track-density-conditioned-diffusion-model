"""
HCP数据集加载脚本
训练集结构: hr (9000, 112, 144), lr (9000, 56, 72), fibre_density (9000, 112, 144) - 扁平结构
测试集结构: {subject: {hr: (30, 30, 112, 144), lr: (30, 30, 56, 72), fibre_density: (30, 112, 144)}} - 有subject分组
"""

import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    HCP数据集训练集
    数据结构: hr (9000, 112, 144), lr (9000, 56, 72), fibre_density (9000, 112, 144)
    每个样本是单通道数据
    """
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        
        with h5py.File(self.h5_file, 'r') as f:
            self.length = f['hr'].shape[0]
            print(f"TrainDataset: 加载 {h5_file}")
            print(f"  hr shape: {f['hr'].shape}")
            print(f"  lr shape: {f['lr'].shape}")
            print(f"  fibre_density shape: {f['fibre_density'].shape}")

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx:idx+1, :, :].astype(np.float32)  # (1, 56, 72)
            hr = f['hr'][idx:idx+1, :, :].astype(np.float32)  # (1, 112, 144)
            fibre_density = f['fibre_density'][idx:idx+1, :, :].astype(np.float32)  # (1, 112, 144)
            return lr, hr, fibre_density

    def __len__(self):
        return self.length


class EvalDataset(Dataset):
    """
    HCP数据集测试集
    测试集有subject分组: {subject: {hr: (30, 30, 112, 144), lr: (30, 30, 56, 72), fibre_density: (30, 112, 144)}}
    每个subject有30个z切片，每个切片30个通道
    """
    def __init__(self, h5_file, number=None):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.number = number
        
        with h5py.File(self.h5_file, 'r') as f:
            # 检查是否有subject分组
            if number is not None and number in f.keys():
                # 有subject分组的情况
                self.has_subjects = True
                self.length = f[number]['hr'].shape[0]  # z切片数量
                print(f"EvalDataset: 加载 {h5_file}, subject: {number}")
                print(f"  hr shape: {f[number]['hr'].shape}")
                print(f"  lr shape: {f[number]['lr'].shape}")
                print(f"  fibre_density shape: {f[number]['fibre_density'].shape}")
            else:
                # 扁平结构的情况
                self.has_subjects = False
                self.length = f['hr'].shape[0]
                print(f"EvalDataset: 加载 {h5_file} (扁平结构)")
                print(f"  hr shape: {f['hr'].shape}")

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            if self.has_subjects:
                # 有subject分组：返回30通道数据
                # hr: (30, 30, 112, 144) -> 取idx切片 -> (30, 112, 144)
                # lr: (30, 30, 56, 72) -> 取idx切片 -> (30, 56, 72)
                # fibre_density: (30, 112, 144) -> 取idx切片 -> (112, 144)
                lr = f[self.number]['lr'][idx].astype(np.float32)  # (30, 56, 72)
                hr = f[self.number]['hr'][idx].astype(np.float32)  # (30, 112, 144)
                fibre_density = f[self.number]['fibre_density'][idx].astype(np.float32)  # (112, 144)
                # 添加通道维度
                if fibre_density.ndim == 2:
                    fibre_density = fibre_density[np.newaxis, :, :]  # (1, 112, 144)
            else:
                # 扁平结构：返回单通道数据
                lr = f['lr'][idx:idx+1, :, :].astype(np.float32)  # (1, h, w)
                hr = f['hr'][idx:idx+1, :, :].astype(np.float32)  # (1, H, W)
                fibre_density = f['fibre_density'][idx:idx+1, :, :].astype(np.float32)  # (1, H, W)
            return lr, hr, fibre_density

    def __len__(self):
        return self.length
