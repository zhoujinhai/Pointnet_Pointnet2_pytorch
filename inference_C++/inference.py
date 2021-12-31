import os
import torch
import numpy as np
from net2onnx import GetModel


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


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
    @param2: use_normal, 数据是否包含法向量
    @param3: use_gpu, 是否使用GPU
    """
    def __init__(self, model_path="./models/weights.pth", use_normal=False, use_gpu=True):
        self.device = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self.use_normal = use_normal
        self.num_cls = 1
        self.num_part = 2
        self.net = GetModel(self.num_part, normal_channel=use_normal, num_categories=self.num_cls).to(self.device)
        self.label = torch.tensor([[0]]).long().to(self.device)
        self.cate = to_categorical(self.label, self.num_cls)
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net = self.net.eval()

    """
    desc: deal points data
    @param1: data_path, 数据路径
    """
    def process_data(self, input_data, is_file=True):
        data = None
        if is_file:
            ext = os.path.splitext(input_data)[-1]
            if ext == ".txt":
                data = np.loadtxt(input_data).astype(np.float32)
            elif ext == ".pcd":
                data = np.loadtxt(input_data, skiprows=10).astype(np.float32)
        else:
            # may be string data
            input = []
            lines = input_data.split("\n")
            for line in lines:
                if line is "":
                    continue
                splitted_line = line.split(" ")
                try:
                    input.append([float(v) for v in splitted_line])
                except:
                    continue
            data = np.asarray(input, dtype=np.float32)

        if data is None:
            return None, None

        if self.use_normal:
            point_set = data[:, 0:6]
        else:
            point_set = data[:, 0:3]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(range(0, len(point_set)), len(point_set), replace=False)
        point_set = point_set[choice, :]

        point_set = np.expand_dims(point_set, axis=0)
        return torch.from_numpy(point_set), choice

    """
    desc: inference points
    @param1: points, 点集
    """
    def inference(self, data_path, is_file=True):
        points, choices = self.process_data(data_path, is_file)
        if points is None:
            return []
        points = points.float().to(self.device)
        points = points.transpose(2, 1)

        with torch.no_grad():
            seg_pred = self.net(points, self.cate)
            cur_pred_val = seg_pred.cpu().data.numpy()

            cur_pred_val_logits = cur_pred_val
            print("&&&&&&&&&res: ", cur_pred_val_logits)
            result = np.argmax(cur_pred_val_logits, 2)
            idx = choices[np.where(result == 1)[1]]
            return idx


def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3]

    colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]
    labels = data[:, label_cls]  # 最后一列为标签列
    diff_label = np.unique(labels)
    print("res_label: ", diff_label)
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
    model_path = "../log/part_seg/pointnet2_part_seg_msg_add_data/checkpoints/best_model.pth"  #
    data_path = r"D:\Debug_dir\news_data\pcd_label_normal\bankou (1)_minCruv.pcd"
    inference = InferenceClass(model_path, use_normal=False, use_gpu=True)
    res = inference.inference(data_path)
    print(res, len(res))
    # show result
    data = np.loadtxt(data_path, skiprows=10).astype(np.float32)
    label = [0] * len(data)
    for r in res:
        label[r] = 1
    show_data = np.c_[data, np.asarray(label)]
    show_pcl_data(show_data)
