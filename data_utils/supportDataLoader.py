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


class SemSegSupportDataset(Dataset):
    def __init__(self, root='/data/pcd_dental_texture', npoints=20000, split='train', normal_channel=False, shuffle=False, n_class=2, f_cols=3):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.f_cols = f_cols
        print("load data!")
        npy_files = glob.glob(os.path.join(self.root, "*.npy"))
        # print("all_data: ", len(npy_files))
        # rot = ["X15", "X30", "X45", "X60", "X75", "X90", "Y15", "Y30", "Y45", "Y60", "Y75", "Y90"]
        # npy_files = [file for file in npy_files if file[-13:-10] not in rot]
        n_data = len(npy_files)
        print("ori_data: ", len(npy_files))
        if split != "test":
            label_weights = np.zeros(n_class)
            for npy_file in npy_files:
                data = np.load(npy_file).astype(np.float32) 
                data = data[data[:, 5] < 0]  # nz < 0
                # print(npy_file, data.shape)
                labels = data[:, -1].astype(np.int32)
                labels[labels < 1] = 0
                labels[labels == 1] = 1
                # labels[labels == 2] = 2
                # labels[labels == 3] = 3
                tmp, _ = np.histogram(labels, range(n_class + 1))
                label_weights += tmp
            print("label_weights: ", label_weights)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
            print("label_weights: ", self.label_weights)

        if shuffle:
            random.shuffle(npy_files)  # 随机打乱
        # self.data_path = []
        # if split == "train":
        #     self.data_path = npy_files[: int(0.8 * n_data) + 1]
        # elif split == "train_val":
        #     self.data_path = npy_files[: int(0.9 * n_data) + 1]
        # elif split == "test":
        #     if shuffle:
        #         self.data_path = npy_files[: int(0.1 * n_data) + 1]
        #     else:
        #         self.data_path = npy_files[int(0.9 * n_data):]
        # else:
        #     self.data_path = npy_files
        self.data_path = npy_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1, 2, 3]}

    def __getitem__(self, index):
        fn = self.data_path[index] 
        data = np.load(fn).astype(np.float32)
        # print(data.shape)
        data = data[data[:, 5] < 0]  # remove nz > 0
        # print("---", data.shape)
        np.random.shuffle(data)
        data = data[:self.npoints, :]
        # print(fn, data.shape)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:self.f_cols]

        label = data[:, -1].astype(np.int32)
        label[label < 1] = 0
        label[label == 1] = 1
        ori_xyz = point_set[:, 0:3]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) 
        point_set[:, 6: self.f_cols]  = pc_normalize(point_set[:, 6:self.f_cols])  
        # print("after_pc_normalize:", point_set.shape, label.shape)
        
        cls_tooth = np.array([0])   # just has one cls: tooth , for part_seg
        return point_set, label, cls_tooth, ori_xyz

    def __len__(self):
        return len(self.data_path)


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
    ori_xyz_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        label = batch_data[batch][1] 
        cls = batch_data[batch][2]
        ori_xyz = batch_data[batch][3]

        choice = np.random.choice(range(0, len(data)), min_len, replace=False)
        data = data[choice, :]
        label = label[choice]
        ori_xyz = ori_xyz[choice, :]

        data_list.append(data)
        cls_list.append(cls)
        label_list.append(label)
        ori_xyz_list.append(ori_xyz)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    ori_xyz_tensor = torch.tensor(ori_xyz_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor, cls_tensor, ori_xyz_tensor)
    return data_copy



class RegSupportDataset(Dataset):
    def __init__(self, root='/data/pcd_dental_texture', npoints=20000, split='train', normal_channel=False, shuffle=False, n_class=2, f_cols=3):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.f_cols = f_cols
        print("load data!")
        npy_files = glob.glob(os.path.join(self.root, "*.npy"))
        # print("all_data: ", len(npy_files))
        # rot = ["X15", "X30", "X45", "X60", "X75", "X90", "Y15", "Y30", "Y45", "Y60", "Y75", "Y90"]
        # npy_files = [file for file in npy_files if file[-13:-10] not in rot]
        n_data = len(npy_files)
        print("ori_data: ", len(npy_files))
        if split != "test":
            label_weights = np.zeros(n_class)
            for npy_file in npy_files:
                data = np.load(npy_file).astype(np.float32) 
                data = data[data[:, 5] < 0]  # nz < 0
                # print(npy_file, data.shape)
                anchor_labels = data[:, -1].astype(np.int32)
                labels = data[:, -2].astype(np.int32)
                labels[(labels < 3) | (anchor_labels == 0)] = 0
                labels[(labels == 3) & (anchor_labels == 1)] = 1
                # labels[labels == 2] = 2
                # labels[labels == 3] = 3
                tmp, _ = np.histogram(labels, range(n_class + 1))
                label_weights += tmp
            print("label_weights: ", label_weights)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
            print("label_weights: ", self.label_weights)

        if shuffle:
            random.shuffle(npy_files)  # 随机打乱
        # self.data_path = []
        # if split == "train":
        #     self.data_path = npy_files[: int(0.8 * n_data) + 1]
        # elif split == "train_val":
        #     self.data_path = npy_files[: int(0.9 * n_data) + 1]
        # elif split == "test":
        #     if shuffle:
        #         self.data_path = npy_files[: int(0.1 * n_data) + 1]
        #     else:
        #         self.data_path = npy_files[int(0.9 * n_data):]
        # else:
        #     self.data_path = npy_files
        self.data_path = npy_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1]} # , 2, 3

    def __getitem__(self, index):
        fn = self.data_path[index] 
        data = np.load(fn).astype(np.float32)
        # print(data.shape)
        data = data[data[:, 5] < 0]  # remove nz > 0
        # print("---", data.shape)
        np.random.shuffle(data)
        data = data[:self.npoints, :]
        # print(fn, data.shape)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:self.f_cols]

        anchor_labels = data[:, -1].astype(np.int32)
        labels = data[:, -2].astype(np.int32)
        labels[(labels < 3) | (anchor_labels == 0)] = 0
        labels[(labels == 3) & (anchor_labels == 1)] = 1
        ori_xyz = point_set[:, 0:3]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) 
        point_set[:, 6: self.f_cols]  = pc_normalize(point_set[:, 6:self.f_cols])  
        # print("after_pc_normalize:", point_set.shape, label.shape)
        
        cls_tooth = np.array([0])   # just has one cls: tooth , for part_seg

        gt_coords = point_set[:, 0:3][labels == 1]
        gt_cnt = np.float32(len(gt_coords))
        return point_set, gt_coords, gt_cnt, cls_tooth

    def __len__(self):
        return len(self.data_path)


def my_collate_fn_reg(batch_data):
    """
    descriptions: 对齐批量数据维度, [(data, label),(data, label)...]转化成([data, data...],[label,label...])
    :param batch_data:  list，[(data, label),(data, label)...]
    :return: tuple, ([data, data...],[label,label...])
    """
    batch_data.sort(key=lambda x: len(x[0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    gt_coords_list = []
    gt_cnt_list = []
    label_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        coords = batch_data[batch][1] 
        cnt = batch_data[batch][2]
        label = batch_data[batch][3]

        choice = np.random.choice(range(0, len(data)), min_len, replace=False)
        data = data[choice, :]
        label = label[choice]
        ori_xyz = ori_xyz[choice, :]

        data_list.append(data)
        cls_list.append(cls)
        label_list.append(label)
        ori_xyz_list.append(ori_xyz)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    ori_xyz_tensor = torch.tensor(ori_xyz_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor, cls_tensor, ori_xyz_tensor)
    return data_copy


if __name__ == "__main__":
    dataset = SemSegSupportDataset(root=r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd", split='test', normal_channel=False)   # /home/heygears/jinhai_zhou/data/pcd_with_label_normal
    print(len(dataset))
    # points, label, target = dataset[8]
    # print(points[0], type(points[0]), type(points[0][0]), label, type(label), target, type(target))
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn_sem)
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
