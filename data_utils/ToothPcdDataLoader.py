# *_*coding:utf-8 *_*
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='/home/heygears/jinhai_zhou/data/pcd_dental_texture', npoints=8192, split='train', normal_channel=False, shuffle=True):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel

        pcd_files = glob.glob(os.path.join(self.root, "*.pcd"))   # pcd data format: x y z label nId / x y z ...(other feature) label nId
        if split.find("test") == -1:
            label_weights = np.zeros(2)
            for pcd_file in pcd_files:
                data = np.loadtxt(pcd_file, skiprows=10).astype(np.float32)
                labels = data[:, -2].astype(np.int32)
                labels[labels > 0] = 1
                tmp, _ = np.histogram(labels, range(3))
                label_weights += tmp
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
        # print(self.label_weights, type(self.label_weights))
        n_data = len(pcd_files)
        if shuffle:
            random.shuffle(pcd_files)  # 随机打乱
        self.data_path = []
        if split == "train":
            self.data_path = pcd_files[: int(0.85 * n_data) + 1]
        elif split == "train_val":
            self.data_path = pcd_files[: int(0.95 * n_data) + 1]
        elif split == "test":
            if shuffle:
                self.data_path = pcd_files[: int(0.1 * n_data) + 1]
            else:
                self.data_path = pcd_files[int(0.9 * n_data):]
        else:
            self.data_path = pcd_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 10

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.data_path[index]
            cls = np.array([0])
            data = np.loadtxt(fn, skiprows=10).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -2].astype(np.int32)
            seg[seg > 0] = 1
            diff_label = np.unique(seg)
            # print("diff_label: ", diff_label)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # if len(seg) >= self.npoints:
        #     choice = np.random.choice(range(0, len(seg)), self.npoints, replace=False)
        # else:
        #     choice = np.random.choice(range(0, len(seg)), self.npoints, replace=True)
        # # resample
        # point_set = point_set[choice, :]
        # seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.data_path)


class SemSegDataset(Dataset):
    def __init__(self, root='/home/heygears/jinhai_zhou/data/pcd_dental_texture', npoints=8192, split='train', normal_channel=False, shuffle=True):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel

        pcd_files = glob.glob(os.path.join(self.root, "*.pcd"))
        n_data = len(pcd_files)

        if split != "test":
            label_weights = np.zeros(2)
            for pcd_file in pcd_files:
                data = np.loadtxt(pcd_file, skiprows=10).astype(np.float32)
                labels = data[:, -2].astype(np.int32)
                labels[labels > 0] = 1
                tmp, _ = np.histogram(labels, range(3))
                label_weights += tmp
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
        # print("label_weights: ", self.label_weights)

        if shuffle:
            random.shuffle(pcd_files)  # 随机打乱
        self.data_path = []
        if split == "train":
            self.data_path = pcd_files[: int(0.8 * n_data) + 1]
        elif split == "train_val":
            self.data_path = pcd_files[: int(0.9 * n_data) + 1]
        elif split == "test":
            if shuffle:
                self.data_path = pcd_files[: int(0.1 * n_data) + 1]
            else:
                self.data_path = pcd_files[int(0.9 * n_data):]
        else:
            self.data_path = pcd_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1]}

    def __getitem__(self, index):
        fn = self.data_path[index]
        # print(fn)
        data = np.loadtxt(fn, skiprows=10).astype(np.float32)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]

        label = data[:, -2].astype(np.int32)
        label[label > 0] = 1

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # print("after_pc_normalize:", point_set)
        return point_set, label

    def __len__(self):
        return len(self.data_path)


def my_collate_fn(batch_data):
    """
    descriptions: 对齐批量数据维度, [(data, label),(data, label)...]转化成([data, data...],[label,label...])
    :param batch_data:  list，[(data, label),(data, label)...]
    :return: tuple, ([data, data...],[label,label...])
    """
    batch_data.sort(key=lambda x: len(x[0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    cls_list = []
    label_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        cls = batch_data[batch][1]
        label = batch_data[batch][2]

        choice = np.random.choice(range(0, len(data)), min_len, replace=False)
        data = data[choice, :]
        label = label[choice]

        data_list.append(data)
        cls_list.append(cls)
        label_list.append(label)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, cls_tensor, label_tensor)
    return data_copy


def my_collate_fn_sem(batch_data):
    """
    descriptions: 对齐批量数据维度, [(data, label),(data, label)...]转化成([data, data...],[label,label...])
    :param batch_data:  list，[(data, label),(data, label)...]
    :return: tuple, ([data, data...],[label,label...])
    """
    batch_data.sort(key=lambda x: len(x[0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    label_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        label = batch_data[batch][1]
        choice = np.random.choice(range(0, len(data)), min_len, replace=False)

        data = data[choice, :]
        label = label[choice]
        data_list.append(data)
        label_list.append(label)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy


if __name__ == "__main__":
    dataset = PartNormalDataset(root=r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd", split='test', normal_channel=False)   # /home/heygears/jinhai_zhou/data/pcd_with_label_normal
    print(len(dataset))
    # points, label, target = dataset[8]
    # print(points[0], type(points[0]), type(points[0][0]), label, type(label), target, type(target))
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)
    print(len(dataLoader))
    import provider
    for i_batch, (points, cls, label) in enumerate(dataLoader):
        if i_batch == 0:
            print(points, label)
            print(points.shape)
            np.savetxt(r"D:\Debug_dir\origin.pts", points[0])
            points = points.data.numpy()
            points[:, :, 0:6] = provider.rotate_point_cloud_xyz(points[:, :, 0:6])
            print(points)
            print(points[0].shape)
            np.savetxt(r"D:\Debug_dir\rotated.pts", points[0])