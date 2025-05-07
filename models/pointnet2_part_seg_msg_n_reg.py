import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.optimize import linear_sum_assignment


class PointFeatureAggregator(nn.Module):
    """特征聚合模块（将B×128×N转换为全局特征）"""
    def __init__(self, mode='max'):
        super().__init__()
        self.mode = mode
        
    def forward(self, x):
        """
        输入: 
            x: (B, 128, N) 逐点特征
        输出:
            global_feat: (B, 256) 聚合后的全局特征
        """
        if self.mode == 'max':
            # 最大池化 + 均值池化拼接
            max_feat = F.max_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)  # (B, 128)
            mean_feat = F.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)  # (B, 128)
            return torch.cat([max_feat, mean_feat], dim=1)  # (B, 256)
        else:
            raise ValueError(f"不支持的聚合模式: {self.mode}")
        


class ModifiedRegressionHead(nn.Module):
    def __init__(self, in_dim=256, max_points=1000):
        super().__init__()
        self.max_points = max_points
        self.fc_layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),  # 替换BatchNorm
            nn.ReLU(),
            nn.Linear(512, max_points*3)
        )

    def forward(self, x):  
        return self.fc_layers(x).view(-1, self.max_points, 3)
    

class ModifiedRegressionHead2(nn.Module):
    def __init__(self, in_dim=256, max_points=1000):
        super().__init__()
        self.max_points = max_points
        self.coord_generator = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, max_points*3)
        )
        
        # 引入注意力权重生成器
        self.attn_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_points),
            nn.Sigmoid()
        )

    def forward(self, global_feat, seg_mask):
        """
        global_feat : 全局特征 [B, in_dim]
        seg_mask    : 分类有效掩码 [B, N]
        """
        # 生成原始坐标预测
        raw_coords = self.coord_generator(global_feat).view(-1, self.max_points, 3)  # [B, M, 3]
        
        # 生成注意力权重（基于有效点密度）
        attn_weights = self.attn_layer(global_feat)  # [B, M]
        
        # 获取每批有效点数
        valid_counts = seg_mask.sum(dim=1).float()  # [B]
        
        # 动态调整预测坐标
        scaled_coords = raw_coords * attn_weights.unsqueeze(-1) * (valid_counts / self.max_points).view(-1,1,1)
        
        return scaled_coords
    

class ModifiedCountHead(nn.Module):
    """适配新特征维度的数量预测头"""
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, 256)
        return self.fc(x)

class ModifiedExistHead(nn.Module):
    """存在性预测头（基于逐点特征）"""
    def __init__(self, point_feat_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(point_feat_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, 128, N)
        return self.conv_layers(x).squeeze(1)  # (B, N)
    

class get_model1(nn.Module):
    def __init__(self, num_classes, normal_channel=False, num_categories=16, additional_channel = 0, max_points = 1000):
        super(get_model1, self).__init__()
        self.num_categories = num_categories
        if additional_channel <= 0:
            if normal_channel:
                additional_channel = 3
            else:
                additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])  # 1024  2048  2304
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])                                  # 256   512   768
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        hidden_dim = 128
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + self.num_categories + additional_channel, mlp=[hidden_dim, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)   # B, hidden_dim, N
       
        # 特征聚合模块
        self.aggregator = PointFeatureAggregator(mode='max')
        
        # 回归与数量预测头
        self.reg_head = ModifiedRegressionHead(in_dim=256, max_points=max_points)
        self.count_head = ModifiedCountHead(in_dim=256)
        
        # 存在性预测头（基于逐点特征）
        self.exist_head = ModifiedExistHead(point_feat_dim=128)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) 
        cls_label_one_hot = cls_label.view(B, self.num_categories, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)
        
        # 全局特征聚合
        global_feat = self.aggregator(feat)  # (B, 256)
        # print("global_feat: ", global_feat.shape)
        
        # 回归预测
        coords = self.reg_head(global_feat)  # (B, M, 3)
        # print("coords.shape", coords.shape)
        count = self.count_head(global_feat)  # (B, 1)
        
        # 存在性预测（基于逐点特征）
        exist_probs = self.exist_head(feat)  # (B, N)
        
        return coords, exist_probs, count


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False, num_categories=16, additional_channel = 0, max_points = 1000):
        super(get_model, self).__init__()
        self.num_categories = num_categories
        if additional_channel <= 0:
            if normal_channel:
                additional_channel = 3
            else:
                additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])  # 1024  2048  2304
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])                                  # 256   512   768
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        hidden_dim = 128
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + self.num_categories + additional_channel, mlp=[hidden_dim, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1) 
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead=1, dim_feedforward=512)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)
        # 动态点生成器
        self.query_embed = nn.Embedding(max_points, hidden_dim)

        # 坐标预测头
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # 存在概率头
        self.exist_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 数量预测头
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) 
        cls_label_one_hot = cls_label.view(B, self.num_categories, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)
        
        feat = feat.permute(2, 0, 1)  # (N, B, hidden_dim)
        # Transformer编码
        memory = self.transformer(feat)  # (N, B, hidden_dim)
        
        # 生成候选点
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (M, B, hidden_dim)
        decoded = self.transformer(query, memory)  # (M, B, hidden_dim)
        decoded = decoded.permute(1, 0, 2)  # (B, M, hidden_dim)
        
        # 预测输出
        coords = self.coord_head(decoded)  # (B, M, 3)
        exist_probs = self.exist_head(decoded).squeeze(-1)  # (B, M)
        count_pred = self.count_head(feat.permute(1, 2, 0))  # (B, 1) 
        
        return coords, exist_probs, count_pred


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight=None):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
    

class MultiTaskLoss(nn.Module):
    def __init__(self, coord_weight=1.0, exist_weight=1.0, count_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.exist_weight = exist_weight
        self.count_weight = count_weight
        
    def hungarian_loss(self, pred_coords, exist_probs, gt_coords, weight=None):
        B = pred_coords.shape[0]
        total_loss = 0.0
        
        for b in range(B):
            # 计算代价矩阵
            # print("pred_coords", pred_coords.shape,  "gt_coords: ", gt_coords.shape)
            cost_matrix = torch.cdist(pred_coords[b], gt_coords[b])  # (M, K)
            
            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            # 坐标损失
            coord_loss = F.smooth_l1_loss(pred_coords[b, row_ind], gt_coords[b][col_ind])
            
            # 存在性损失
            existence_mask = torch.zeros_like(exist_probs[b])
            existence_mask[row_ind] = 1.0
            exist_loss = F.binary_cross_entropy(exist_probs[b], existence_mask, weight=weight[0])
            
            total_loss += self.coord_weight * coord_loss + self.exist_weight * exist_loss
            
        return total_loss / B
    
    def forward(self, pred_coords, exist_probs, count_pred, gt_coords, gt_counts, weight=None):
        # 坐标和存在性损失
        loss_coord_exist = self.hungarian_loss(pred_coords, exist_probs, gt_coords, weight)
        
        # 数量回归损失
        # print("count_pred: ", count_pred, gt_counts)
        loss_count = F.mse_loss(count_pred.squeeze(-1), gt_counts.float())
        
        # 总损失
        total_loss = loss_coord_exist + self.count_weight * loss_count
        
        return total_loss
    

class get_model2(nn.Module):
    def __init__(self, num_classes, normal_channel=False, num_categories=16, additional_channel = 0, max_points = 1000):
        super(get_model2, self).__init__()
        self.num_categories = num_categories
        if additional_channel <= 0:
            if normal_channel:
                additional_channel = 3
            else:
                additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])  # 1024  2048  2304
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])                                  # 256   512   768
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        hidden_dim = 128
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + self.num_categories + additional_channel, mlp=[hidden_dim, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # 特征聚合模块
        self.aggregator = PointFeatureAggregator(mode='max')
        
        # 回归与数量预测头
        self.reg_head = ModifiedRegressionHead(in_dim=256, max_points=max_points)
        self.count_head = ModifiedCountHead(in_dim=256)

        # 添加冻结方法
        self._freeze_backbone()

    def _freeze_backbone(self):
        """冻结所有Backbone相关参数"""
        backbone_modules = [
            'sa1', 'sa2', 'sa3',   # 特征提取层
            'fp3', 'fp2', 'fp1',   # 特征传播层
            'conv1', 'bn1', 'drop1', 'conv2'  # 原始分割头
        ]
        for name, param in self.named_parameters():
            if any([name.startswith(m) for m in backbone_modules]):
                param.requires_grad = False
                
        # 可选：设置BatchNorm为评估模式
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) and any([name.startswith(m) for m in backbone_modules]):
                m.eval()

    def unfreeze_layers(self, layer_names):
        """解冻指定层"""  # # 使用示例：解冻最后两个特征传播层 model.unfreeze_layers(['fp2', 'fp3'])
        for name, param in self.named_parameters():
            if any([name.startswith(ln) for ln in layer_names]):
                param.requires_grad = True

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) 
        cls_label_one_hot = cls_label.view(B, self.num_categories, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        seg_out = x.permute(0, 2, 1)

        # # 获取分类结果与掩码
        # seg_probs = torch.exp(seg_out)  # [B, N, C]
        # valid_mask = seg_probs.argmax(dim=2) > 0  # [B, N]
        
        # # 全局特征聚合（加入掩码信息）
        # masked_feat = feat * valid_mask.unsqueeze(1).float()  # [B, 128, N]
        # global_feat = self.aggregator(masked_feat)  # [B, 256]
        
        # # 回归预测
        # coords = self.reg_head(global_feat, valid_mask)  # [B, M, 3]
        
        # 全局特征聚合
        global_feat = self.aggregator(feat)  # (B, 256)
        # print("global_feat: ", global_feat.shape)
        
        # 回归预测
        coords = self.reg_head(global_feat)  # (B, M, 3)
        # print("coords.shape", coords.shape)
        count = self.count_head(global_feat)  # (B, 1) 

        return coords, seg_out, count
    

class MultiTaskLoss2(nn.Module):
    def __init__(self, coord_weight=2.0, count_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.count_weight = count_weight
        
    def hungarian_loss(self, pred_coords, exist_probs, gt_coords, weight=None):
        B = pred_coords.shape[0]
        total_loss = 0.0
        
        for b in range(B):
            # 计算代价矩阵
            # print("pred_coords", pred_coords.shape,  "gt_coords: ", gt_coords.shape)
            cost_matrix = torch.cdist(pred_coords[b], gt_coords[b])  # (M, K)
            
            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            # 坐标损失
            coord_loss = F.smooth_l1_loss(pred_coords[b, row_ind], gt_coords[b][col_ind])
            
            total_loss += self.coord_weight * coord_loss 
        return total_loss / B
    
    def forward(self, pred_coords, exist_probs, count_pred, gt_coords, gt_counts, weight=None):
        # 坐标损失
        coord_exist = self.hungarian_loss(pred_coords, exist_probs, gt_coords, weight)
        
        # 数量回归损失
        # print("count_pred: ", count_pred, gt_counts)
        loss_count = F.mse_loss(count_pred.squeeze(-1), gt_counts.float())
        
        # 总损失
        total_loss = coord_exist + self.count_weight * loss_count
        
        return total_loss
