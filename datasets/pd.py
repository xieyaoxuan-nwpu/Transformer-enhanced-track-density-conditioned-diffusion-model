import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    PD数据集训练集
    数据结构: hr (4500, 88, 88), lr (4500, 44, 44), fibre_density (4500, 88, 88)
    每个样本是单通道数据
    """
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        
        with h5py.File(self.h5_file, 'r') as f:
            self.length = f['hr'].shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx:idx+1, :, :]  # (1, 44, 44)
            hr = f['hr'][idx:idx+1, :, :]  # (1, 88, 88)
            fibre_density = f['fibre_density'][idx:idx+1, :, :]  # (1, 88, 88)
            return lr, hr, fibre_density

    def __len__(self):
        return self.length


class EvalDataset(Dataset):
    """
    PD数据集测试集
    假设测试集也是类似的扁平结构
    如果测试集有subject分组，需要相应调整
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
                self.length = f[number]['hr'].shape[0]
            else:
                # 扁平结构的情况
                self.has_subjects = False
                self.length = f['hr'].shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            if self.has_subjects:
                # 有subject分组：返回30通道数据
                lr = f[self.number]['lr'][idx]  # (30, h, w)
                hr = f[self.number]['hr'][idx]  # (30, H, W)
                fibre_density = f[self.number]['fibre_density'][idx]  # (1, H, W)
            else:
                # 扁平结构：返回单通道数据
                lr = f['lr'][idx:idx+1, :, :]  # (1, h, w)
                hr = f['hr'][idx:idx+1, :, :]  # (1, H, W)
                fibre_density = f['fibre_density'][idx:idx+1, :, :]  # (1, H, W)
            return lr, hr, fibre_density

    def __len__(self):
        return self.length
