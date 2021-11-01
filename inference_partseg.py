import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2704, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class InferenceClass(object):
    """
    desc: init inference class
    @param1: model_path, 模型文件路径
    @param2: n_cls, 类别个数
    @param3: use_gpu, 是否使用GPU
    """
    def __init__(self, model_name, model_path, label=0, num_cls=16, num_part=50, use_normal=False, use_gpu=True):
        self.num_cls = num_cls
        self.use_normal = use_normal
        self.device = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        model = importlib.import_module(model_name)  # sys.path.append()
        self.net = model.get_model(num_part, normal_channel=use_normal, num_categories=num_cls).to(self.device)
        self.label = torch.from_numpy(np.array([[label]])).long().to(self.device)
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net = self.net.eval()

    """
    desc: deal points data
    @param1: data_path, 数据路径
    """
    def process_data(self, data_path):
        ext = os.path.splitext(data_path)[-1]
        data = None
        if ext == ".txt":
            data = np.loadtxt(data_path).astype(np.float32)
        elif ext == ".pcd":
            data = np.loadtxt(data_path, skiprows=10).astype(np.float32)

        if data is None:
            return

        if not self.use_normal:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = np.expand_dims(point_set, axis=0)
        return torch.from_numpy(point_set)

    """
    desc: inference points
    @param1: points, 点集
    """
    def inference(self, data_path):
        points = self.process_data(data_path)
        points = points.float().to(self.device)
        points = points.transpose(2, 1)

        with torch.no_grad():
            seg_pred, _ = self.net(points, to_categorical(self.label, self.num_cls))
            cur_pred_val = seg_pred.cpu().data.numpy()

            cur_pred_val_logits = cur_pred_val

            cur_pred_val = np.argmax(cur_pred_val_logits, 2)
            return cur_pred_val[0]


def show_pcl_data(data, label_cls=-1):
    import vedo
    colours = ["red", "grey", "blue", "brown", "yellow", "green", "black", "pink"]
    labels = data[:, label_cls]  # 最后一列为标签列
    diff_label = np.unique(labels)
    print("diff_label: ", diff_label)
    group_points = []
    for label in diff_label:
        point_group = points[labels == label]
        group_points.append(point_group)

    show_pts = []
    for i, point in enumerate(group_points):
        pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


if __name__ == '__main__':
    classes = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6,
               'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12,
               'Rocket': 13, 'Skateboard': 14, 'Table': 15}
    categories = {'02691156': 'Airplane', '02773838': "Bag", '02954340': "Cap", '02958343': "Car",
                  '03001627': "Chair", '03261776': "Earphone", '03467517': "Guitar", '03624134': "Knife",
                  '03636649': "Lamp", '03642806': "Laptop", '03790512': "Motorbike", '03797390': "Mug",
                  '03948459': "Pistol", '04099429': "Rocket", '04225987': "Skateboard", '04379243': "Table"}
    model_name = "pointnet_part_seg"
    model_path = "log/part_seg/pointnet_part_seg_test/checkpoints/best_model.pth"
    data_path = r"D:\Documents\Downloads\shapenetcore_partanno_segmentation_benchmark_v0_normal/03001627/355fa0f35b61fdd7aa74a6b5ee13e775.txt"
    cate_id = os.path.basename(os.path.dirname(data_path))
    cate = categories[cate_id]
    label = classes[cate]
    inference = InferenceClass(model_name, model_path, label=label, num_cls=16, num_part=50, use_normal=False, use_gpu=True)
    res = inference.inference(data_path)

    points = np.loadtxt(data_path)[:, :3]
    data = np.c_[points, res]
    show_pcl_data(data)
