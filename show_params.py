from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
            x = torch.bmm(x, trans)
            x = torch.cat([x, feature], dim=2)
        else:
            x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class get_model_cls(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model_cls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_model_seg(nn.Module):
    def __init__(self, part_num=50, normal_channel=False, num_categories=16):
        super(get_model_seg, self).__init__()
        self.num_categories = num_categories
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4928 + self.num_categories, 256, 1)  # 4928 is concat
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):

        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            point_cloud = torch.cat([point_cloud, feature], dim=2)
        else:
            point_cloud = torch.bmm(point_cloud, trans)

        point_cloud = point_cloud.transpose(2, 1)
        out1 = F.relu(self.bn1(self.conv1(point_cloud)))   # n * 64
        out2 = F.relu(self.bn2(self.conv2(out1)))          # n * 128
        out3 = F.relu(self.bn3(self.conv3(out2)))          # n * 128
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)  # n * 128

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))  # n * 512
        out5 = self.bn5(self.conv5(out4))                  # n * 2048
        out_max = torch.max(out5, 2, keepdim=True)[0]      # 1 * 2048
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048+self.num_categories, 1).repeat(1, 1, N)   # n * (2048 + 16), 16 is class number
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)  # n * 4944, pointNet paper is cat with expand, out4, net_transformed, out3, out2, out1 ==> n * 3024
        net = F.relu(self.bns1(self.convs1(concat)))        # n * 256
        net = F.relu(self.bns2(self.convs2(net)))           # n * 256
        net = F.relu(self.bns3(self.convs3(net)))           # n * 128
        net = self.convs4(net)                              # n * 50, 50 is part number
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)  # [B, N, 50]

        return net, trans_feat


if __name__ == "__main__":
    net = get_model_seg()
    net = net.to("cuda")
    summary(net, input_size=[(3, 2000), (1, 16)], batch_size=1, device="cuda")
