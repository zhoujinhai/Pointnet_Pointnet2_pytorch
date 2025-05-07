import argparse
import os
import torch
import logging
import sys
import importlib
import numpy as np
from data_utils.ToothPcdDataLoader import PartNormalDataset, my_collate_fn, SemSegDataset, my_collate_fn_sem
import time


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


def nms_3d_point_cloud(points, labels, scores, radius=1.0, cls = 2):
    """
    对三维点云进行NMS过滤
    :param points: (N, 3) 点云坐标[x, y, z]
    :param labels: (N,) 点云标签[0, 1, 2, 3]
    :param scores: (N,) 点云置信度（值越大表示置信度越高）
    :param radius: 抑制半径 
    """
    print("points shape: ", points.shape, labels.shape, scores.shape)
    from scipy.spatial import KDTree
    # 分离前景和背景
    foreground_mask = (labels != 0)
    fg_points = points[foreground_mask]
    fg_labels = labels[foreground_mask]
    fg_scores = scores[foreground_mask]
    original_indices = np.where(foreground_mask)[0]
    print(original_indices.shape)
    # 按类别处理每个前景类
    filtered_indices = []
    for class_id in range(1, cls):
        class_mask = (fg_labels == class_id)
        if not np.any(class_mask):
            continue
        class_points = fg_points[class_mask]
        class_scores = fg_scores[class_mask]
        class_original_indices = original_indices[class_mask]  # 原始数据中的索引

        # 构建KD树加速邻域搜索
        kdtree = KDTree(class_points)
        remain_indices = []
        visited = set()

        # 按置信度降序处理
        sorted_ids = np.argsort(class_scores)[::-1]
        for idx in sorted_ids:
            if idx in visited:
                continue
            remain_indices.append(idx)
            point = class_points[idx]
            neighbors = kdtree.query_ball_point(point, radius)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)

        # 确定被抑制的点（不在remain_indices中的点）的原始索引
        print(class_id, len(remain_indices))
        suppressed_mask = np.ones(len(class_original_indices), dtype=bool)
        suppressed_mask[remain_indices] = False
        suppressed_original = class_original_indices[suppressed_mask]

        # 将被抑制的点设为背景类（标签0）
        labels[suppressed_original] = 0


def select_unique_points(pred_coords, probs, pred_count, points):
    """
    从原始点云中选择与预测坐标最近且不重复的点，按概率降序选取指定数量
    
    参数：
        pred_coords : torch.Tensor 预测坐标 [B, M, 3] 
        probs      : torch.Tensor 原始点概率 [B, N]
        pred_count : torch.Tensor 需要选取的点数 [B, 1]
        points     : torch.Tensor 原始点云坐标 [B, N, 3]
    
    返回：
        selected_points : torch.Tensor 筛选后的点云 [B, K, 3]
    """
    # 压缩批次维度（假设batch_size=1）
    B, M, _ = pred_coords.shape
    _, N, _ = points.shape
    
    # 计算所有预测点到原始点的距离矩阵 [B, M, N]
    dists = torch.cdist(pred_coords, points)
    
    # 存储最终结果的列表
    batch_results = [] 
    for b in range(B):
        # 当前批次数据
        batch_dists = dists[b]          # [M, N]
        batch_probs = probs[b]          # [N]
        current_count = int(pred_count[b].item())
        
        # 1. 找到每个预测点的最近原始点索引 [M]
        _, min_indices = torch.min(batch_dists, dim=1)
        
        # 2. 建立字典记录每个原始点的最大概率
        point_dict = {}
        for pred_idx, orig_idx in enumerate(min_indices):
            orig_idx = orig_idx.item()
            current_prob = batch_probs[orig_idx].item()
            
            # 更新字典：只保留最大概率的预测点
            if orig_idx not in point_dict or current_prob > point_dict[orig_idx][0]:
                point_dict[orig_idx] = (current_prob, pred_idx)
        
        # 3. 提取并排序唯一点
        sorted_items = sorted(point_dict.items(), key=lambda x: -x[1][0])
        
        # 4. 按概率选取前K个点
        selected = [] 
        for idx, (prob, pred_idx) in sorted_items[:current_count]:
            selected.append(points[b, idx]) 
        
        # 转换为Tensor并保持维度
        if len(selected) > 0:
            selected = torch.stack(selected).unsqueeze(0)  # [1, K, 3]
        else:
            selected = torch.zeros(1, 0, 3, device=points.device)
            
        batch_results.append(selected) 
    
    # 合并批次结果
    return torch.cat(batch_results, dim=0)


def select_unique_indices(pred_coords, probs, pred_count, points):
    """
    返回与预测坐标最近的原始点索引，并满足数量要求
    
    参数：
        pred_coords : [B, M, 3] 预测坐标
        probs       : [B, N] 原始点概率
        pred_count  : [B, 1] 需要选取的点数
        points      : [B, N, 3] 原始点云
    
    返回：
        selected_indices : [B, K] 原始点云中的索引（K <= pred_count）
    """
    B, M, _ = pred_coords.shape
    device = pred_coords.device
    
    # 计算距离矩阵 [B, M, N]
    dists = torch.cdist(pred_coords, points)
    
    batch_results = []
    
    for b in range(B):
        current_dists = dists[b]        # [M, N]
        current_probs = probs[b]       # [N]
        current_count = int(pred_count[b].item())
        
        # 找到每个预测点的最近原始点索引 [M]
        _, min_indices = torch.min(current_dists, dim=1)  # [M]
        
        # 建立索引-概率映射，保留最大概率
        index_dict = {}
        for pred_idx, orig_idx in enumerate(min_indices):
            orig_idx = orig_idx.item()
            prob = current_probs[orig_idx].item()
            if orig_idx not in index_dict or prob > index_dict[orig_idx][0]:
                index_dict[orig_idx] = (prob, pred_idx)
        
        # 按概率降序排序
        sorted_indices = sorted(index_dict.items(), 
                              key=lambda x: -x[1][0])  # 按概率降序
        
        # 选取前K个索引
        selected = [idx for idx, _ in sorted_indices[:current_count]]
        
        # 转换为Tensor
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
        batch_results.append(selected_tensor.unsqueeze(0))  # 保持批次维度
        
    return torch.cat(batch_results, dim=0)  # [B, K]


class InferenceClass(object):
    """
    desc: init inference class
    @param1: model_path, 模型文件路径
    @param2: n_cls, 类别个数
    @param3: use_gpu, 是否使用GPU
    """
    def __init__(self, model_name, model_path, model_type="part", use_normal=False, use_gpu=True, num_class=2, channel = 3, num_part = 2):
        self.device = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('cpu')  
        self.use_normal = use_normal
        self.model_type = model_type
        self.num_cls = num_class
        self.channel = channel
        self.max_points = 1000
        if model_type == "part":
            self.num_part = num_part
            model = importlib.import_module(model_name)  # sys.path.append()  
             
            self.net = model.get_model2(self.num_part, normal_channel=use_normal, num_categories=self.num_cls, additional_channel=self.channel-3, max_points=self.max_points).to(self.device)
            self.label = torch.from_numpy(np.array([[0]])).long().to(self.device)
        elif model_type == "sem":
            # if self.use_normal:
            #     channel = 12
            # else:
            #     channel = 3
            model = importlib.import_module(model_name)
            self.net = model.get_model(self.num_cls, self.channel).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model_epoch = checkpoint['epoch'] 
        if 'min_loss' in checkpoint.keys():
            min_loss = checkpoint['min_loss']
            print("min_loss: ", min_loss)
        self.net.load_state_dict(checkpoint['model_state_dict']) 
        print("loaded model!, epoch is :", model_epoch)
        if 'class_avg_iou' in checkpoint.keys():
            best_iou = checkpoint['class_avg_iou']
            # print("load best iou: ", best_iou)
        if 'best_acc' in checkpoint.keys():
            best_acc = checkpoint['best_acc']
            # print("load best acc: ", best_acc)
        self.net = self.net.eval()

    """
    desc: deal points data
    @param1: data_path, 数据路径
    """
    def process_data(self, data_path, bAnchor=False):
        ext = os.path.splitext(data_path)[-1]
        data = None
        ori_normal_data = None
        ori_normal_neg_data = None
        if ext == ".txt":
            data = np.loadtxt(data_path).astype(np.float32)
        elif ext == ".pcd":
            data = np.loadtxt(data_path, skiprows=10).astype(np.float32)
        elif ext == ".npy":
            data = np.load(data_path).astype(np.float32)   
            ori_normal_data = data[data[:, 5] >= 0][:, :self.channel+1 + 1] 
            ori_normal_data[:, -1] = 0  
            ori_normal_neg_data = data[data[:, 5] < 0][:, :self.channel + 1] 
            data = data[data[:, 5] < 0] 
            
        if data is None:
            return 
        if not self.use_normal:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:self.channel]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set[:, 6: self.channel]  = pc_normalize(point_set[:, 6:self.channel])  
         
        # choice = np.random.choice(range(0, len(point_set)), len(point_set), replace=False)
        # print(choice)
        # point_set = point_set[choice, :]
        indices = np.arange(len(point_set))
        np.random.shuffle(indices)
        point_set = point_set[indices]
        point_set = np.expand_dims(point_set, axis=0)
        data = data[indices]
        label = data[:, -1].astype(np.int32)
        if bAnchor:
            anchor_label = data[:, -1].astype(np.int32)
            label = data[:, -2].astype(np.int32) 
            print("anchor size: ", anchor_label[(anchor_label > 0) & (label == 3)].shape)
            return torch.from_numpy(point_set), label, anchor_label, ori_normal_data, ori_normal_neg_data[indices]
        return torch.from_numpy(point_set), label, ori_normal_data, ori_normal_neg_data[indices]
    

    """
    desc: inference points
    @param1: points, 点集
    """
    def inference(self, data_path, use_dataset=True, split="test"):
          
        points, label, anchor_label, ori_normal_data, ori_normal_neg_data = self.process_data(data_path, True)
        points = points.float().to(self.device)
        points = points.transpose(2, 1)
        print("**********")
        with torch.no_grad():
            cls_tooth = torch.Tensor([0])  
            pred_coords, probs, pred_count = self.net(points, to_categorical(cls_tooth.long(), self.num_cls))
            print(pred_coords.shape, probs[0].shape, pred_count, points.shape)
            cnt = pred_count * self.max_points
            
            probs = torch.exp(probs)
            max_values, max_indices = torch.max(probs, dim=2)
            points = points.transpose(2, 1)
            max_values[max_indices == 0] = 0
            print(points.shape, max_values.shape, max_indices.shape) 
            probs = max_values
            selected_indexes = select_unique_indices(pred_coords, probs, cnt, points[:, :, :3])
            print(selected_indexes, selected_indexes.shape)

            masks = torch.zeros(points.shape[0], points.shape[1], 
                      dtype=torch.float32, 
                      device=selected_indexes.device)
            print(masks.shape)
            # 将索引位置设为1（使用scatter_高效实现）
            src = torch.ones_like(selected_indexes, dtype=torch.float32)  # 生成全1的源张量
            masks.scatter_(dim=1, index=selected_indexes, src=src)
 
            index_neg = points[:, :, 5] > 0
            print("index_neg: ", index_neg, index_neg.shape)
            masks[index_neg] = 0  
            masks[max_indices == 0] = 0 
            labels = masks[0].cpu().numpy()
             
            nms_3d_point_cloud(ori_normal_neg_data[:, :3], labels, probs[0].cpu().numpy())
            show_data = np.c_[ori_normal_neg_data, labels] 
            show_data = np.vstack((show_data, ori_normal_data)) 
            show_pcl_data(show_data) 


def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3] 

    colours = ["grey", "red", "blue", "yellow", "brown", "green", "black", "pink"]
    labels = data[:, label_cls]  # 最后一列为标签列
    diff_label = np.unique(labels)
    print("res_label: ", diff_label)
    group_points = []
    group_labels = []
    for label in diff_label:
        point_group = points[labels == label]
        group_points.append(point_group)
        # print(point_group.shape)
        group_labels.append(label)

    show_pts = []
    for i, point in enumerate(group_points):
        pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[int(group_labels[i]) % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


def test_acc():
    data_root = r"/data/support/0218/"
    import glob
    data_paths = glob.glob(os.path.join(data_root, "*.npy"))
    rot = ["X15", "X30", "X45", "X60", "X75", "X90", "Y15", "Y30", "Y45", "Y60", "Y75", "Y90"]
    # data_paths = [file for file in data_paths if file[-13:-10] not in rot]
    
    model_type = "part" if model_name.find("part") != -1 else "sem"
    n_class = 4
    channel = 12
    inference = InferenceClass(model_name, model_path, model_type, use_normal=True, use_gpu=False, num_class=n_class, channel=channel)
    acc = 0.0
    acc2 = 0.0
    acc3 = 0.0
    for idx, data_path in enumerate(data_paths): 
        # data_path = r"/data/support/0218/1ALLY_VS_SET_VSc1_Subsetup8_Mandibular__X15_point.npy" 
        use_datasets = False
        print("data path : ", idx,  data_path)
        res = inference.inference2(data_path, use_datasets, "test_all")
        acc += res[0]
        acc2 += res[1]
        acc3 += res[2]
    print(acc, acc2, acc3, len(data_paths))
    print(acc / len(data_paths), acc2 / len(data_paths), acc3 / len(data_paths))


if __name__ == '__main__':
    np.random.seed(0)
    model_name = "pointnet2_part_seg_msg_n_reg"
    model_path = "log/part_seg/pointnet2_part_seg_msg_support_12_normal_4w_sample_reg2/checkpoints/min_loss_model.pth"
    print("Using Model: ", model_path)
    # data_path = r"/data/support/0217/1ALLY_VS_SET_VSc1_Subsetup27_Maxillar__Y60_point.npy" # r"/data/3D_tooth_seg/20190723LowerJawScan_113999648_181205125648_LowerJawScan_label.npy"  # 
    data_path = r"/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
    # data_path = r"/data/support/consurmer/shouban_kedaya_0_point.npy" 
    model_type = "part" if model_name.find("part") != -1 else "sem"
    n_class = 1
    channel = 12
    num_part = 2
    inference = InferenceClass(model_name, model_path, model_type, use_normal=True, use_gpu=False, num_class=n_class, channel=channel, num_part=num_part)
    use_datasets = False
    print("data path : ", data_path)
    res = inference.inference(data_path, use_datasets, "test_all")
    print(res)

    # # sem_seg
    # model_name = "pointnet2_sem_seg"
    # model_path = "log/sem_seg/pointnet2_seg_msg_support_12_normal_4w_sample/checkpoints/best_model.pth"
    # print("Using Model: ", model_path)
    # # data_path = r"/data/support/0217/1ALLY_VS_SET_VSc1_Subsetup27_Maxillar__Y60_point.npy" # r"/data/3D_tooth_seg/20190723LowerJawScan_113999648_181205125648_LowerJawScan_label.npy"  # 
    # data_path = r"/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
    # # data_path = r"/data/support/downsample/7YLY3_VS_SET_VSc2_Subsetup14_Maxillar_point.npy" 
    # model_type = "part" if model_name.find("part") != -1 else "sem"
    # n_class = 4
    # channel = 12
    # inference = InferenceClass(model_name, model_path, model_type, use_normal=True, use_gpu=False, num_class=n_class, channel=channel)
    # use_datasets = False
    # print("data path : ", data_path)
    # res = inference.inference(data_path, use_datasets, "test_all")
    # print(res)

    
