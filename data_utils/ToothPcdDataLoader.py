# *_*coding:utf-8 *_*
import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=8192, split='train', normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel

        pcd_files = glob.glob(os.path.join(self.root, "*.pcd"))
        n_data = len(pcd_files)
        random.shuffle(pcd_files)  # 随机打乱
        self.data_path = []
        if split == "train":
            self.data_path = pcd_files[: int(0.8 * n_data) + 1]
        elif split == "train_val":
            self.data_path = pcd_files[: int(0.9 * n_data) + 1]
        elif split == "test":
            self.data_path = pcd_files[: int(0.1 * n_data) + 1]
        else:
            self.data_path = pcd_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 2000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.data_path[index]
            cls = [0]
            data = np.loadtxt(fn, skiprows=10).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -2].astype(np.int32)
            seg[seg > 0] = 1
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.data_path)


if __name__ == "__main__":
    dataset = PartNormalDataset(root=r"D:\Debug_dir\pcd_with_label")
    print(len(dataset))
    points, label, target = dataset[8]
    print(points, label, target, len(target))
