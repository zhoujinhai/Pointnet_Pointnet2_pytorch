import argparse
import os
import torch
import logging
import sys
import importlib
import numpy as np
from data_utils.ToothPcdDataLoader import PartNormalDataset, my_collate_fn, SemSegDataset, my_collate_fn_sem


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'toothModel': [0, 1]}
seg_label_to_cat = {}   # {0:toothModel, 1:toothModel}
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
    def __init__(self, model_name, model_path, model_type="part", use_normal=False, use_gpu=True):
        self.device = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self.use_normal = use_normal
        self.model_type = model_type
        if model_type == "part":
            self.num_cls = 1
            self.num_part = 2
            model = importlib.import_module(model_name)  # sys.path.append()
            self.net = model.get_model(self.num_part, normal_channel=use_normal, num_categories=self.num_cls).to(self.device)
            self.label = torch.from_numpy(np.array([[0]])).long().to(self.device)
        elif model_type == "sem":
            self.num_cls = 2
            if self.use_normal:
                channel = 6
            else:
                channel = 3
            model = importlib.import_module(model_name)
            self.net = model.get_model(self.num_cls, channel).to(self.device)

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
        choice = np.random.choice(range(0, len(point_set)), len(point_set), replace=False)
        point_set = point_set[choice, :]

        point_set = np.expand_dims(point_set, axis=0)
        return torch.from_numpy(point_set)

    """
    desc: inference points
    @param1: points, 点集
    """
    def inference(self, data_path, use_dataset=True):
        if self.model_type == "part":
            if use_dataset:
                dataset = PartNormalDataset(root=r"D:\Debug_dir\pcd_with_label_normal", split='test', shuffle=False, normal_channel=self.use_normal)  # /home/heygears/jinhai_zhou/data/pcd_with_label
                dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)
                for i_batch, (data, cls, label) in enumerate(dataLoader):
                        points = data.float().to(self.device)

                        points = points.transpose(2, 1)

                        with torch.no_grad():
                            seg_pred, _ = self.net(points, to_categorical(self.label, self.num_cls))
                            cur_pred_val = seg_pred.cpu().data.numpy()
                            cur_pred_val_logits = cur_pred_val
                            cur_pred_val = np.argmax(cur_pred_val_logits, 2)
                            points = points.transpose(2, 1)
                            show_data = np.c_[points.cpu().numpy()[0], cur_pred_val[0]]
                            show_pcl_data(show_data)
            else:
                points = self.process_data(data_path)
                points = points.float().to(self.device)
                points = points.transpose(2, 1)

                with torch.no_grad():
                    print(points.shape, to_categorical(self.label, 16).shape)
                    seg_pred, _ = self.net(points, to_categorical(self.label, self.num_cls))
                    cur_pred_val = seg_pred.cpu().data.numpy()

                    cur_pred_val_logits = cur_pred_val

                    cur_pred_val = np.argmax(cur_pred_val_logits, 2)
                    points = points.transpose(2, 1)
                    show_data = np.c_[points.cpu().numpy()[0], cur_pred_val[0]]
                    show_pcl_data(show_data)

        elif self.model_type == "sem":
            if use_dataset:
                dataset = SemSegDataset(root=r"D:\Debug_dir\pcd_with_label_normal", split='test', shuffle=False, normal_channel=self.use_normal)  # /home/heygears/jinhai_zhou/data/pcd_with_label
                dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn_sem)
                for i_batch, (data, label) in enumerate(dataLoader):
                    points = data.float().to(self.device)
                    points = points.transpose(2, 1)

                    with torch.no_grad():
                        seg_pred, _ = self.net(points)
                        cur_pred_val = seg_pred.cpu().data.numpy()
                        cur_pred_val_logits = cur_pred_val
                        cur_pred_val = np.argmax(cur_pred_val_logits, 2)
                        points = points.transpose(2, 1)
                        show_data = np.c_[points.cpu().numpy()[0], cur_pred_val[0]]
                        show_pcl_data(show_data)
            else:
                points = self.process_data(data_path)
                points = points.float().to(self.device)
                points = points.transpose(2, 1)

                with torch.no_grad():
                    seg_pred, _ = self.net(points)
                    cur_pred_val = seg_pred.cpu().data.numpy()

                    cur_pred_val_logits = cur_pred_val

                    cur_pred_val = np.argmax(cur_pred_val_logits, 2)

                    points = points.transpose(2, 1)
                    show_data = np.c_[points.cpu().numpy()[0], cur_pred_val[0]]
                    show_pcl_data(show_data)
        else:
            print("please check your model type!")


def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3]

    colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]
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
    model_name = "pointnet2_part_seg_msg"
    model_path = "log/part_seg/pointnet2_part_seg_msg_tooth_weight_1024_normal_1/checkpoints/best_model.pth"
    data_path = r"D:\Debug_dir\pcd_with_label\0824-fangshedaoban-kehushuju (101)_minCruv.pcd"
    model_type = "part" if model_name.find("part") != -1 else "sem"
    inference = InferenceClass(model_name, model_path, model_type, use_normal=True, use_gpu=True)
    use_datasets = True
    res = inference.inference(data_path, use_datasets)