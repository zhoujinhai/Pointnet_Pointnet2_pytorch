import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# torch.ops.load_library("./inference_onnx/fps.dll")
#
#
# def my_fps(g, xyz, npoints):
#     return g.op("my_ops::fps", xyz, npoints)
#
#
# torch.onnx.register_custom_op_symbolic("my_ops::fps", my_fps, 9)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    for i in range(1, len(view_shape)):
        view_shape[i] = 1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint: int):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random choice
    v, idx = xyz[:, :, 0].max(1)
    farthest = idx.type(torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # mask = dist < distance
        # distance[mask] = dist[mask]
        distance = torch.where(dist < distance, dist, distance)
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # mask = sqrdists > radius ** 2
    # group_idx[mask] = N
    temp = torch.ones(group_idx.shape, dtype=torch.long) * N    # (B, S, N)
    group_idx = torch.where(sqrdists > radius ** 2, temp, group_idx)

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    group_idx = torch.where(group_idx == N, group_first, group_idx)
    return group_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    # if points is not None:
    #     new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    # else:
    #     new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(npoint: int, radius: float, nsample: int, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)   # [B, npoint, C]
    # fps_idx = torch.ops.my_ops.fps(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)   # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    # if points is not None:
    #     grouped_points = index_points(points, idx)
    #     new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    # else:
    #     new_points = grouped_xyz_norm
    grouped_points = index_points(points, idx)
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)   # [B, C+D, nsample, npoint]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint: int, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3      # ??? due to points?
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))    # 得到最新采样的点
        # new_xyz = index_points(xyz, torch.ops.my_ops.fps(xyz, S))
        new_points_list = []

        i = 0
        for conv_blocks, bn_blocks in zip(self.conv_blocks, self.bn_blocks):
            radius = self.radius_list[i]
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)     # TODO: square dist can be move the outer of query_ball_point
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for conv, bn in zip(conv_blocks, bn_blocks):
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
            i += 1

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        # print(S, type(S))

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class GetModel(nn.Module):
    def __init__(self, num_classes, normal_channel=False, num_categories=16):
        super(GetModel, self).__init__()
        self.num_categories = num_categories
        # TODO 2021/11/19
        additional_channel = 0
        # if normal_channel:
        #     additional_channel = 3
        # else:
        #     additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=5.0, nsample=256, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + self.num_categories + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        # TODO 2021/11/19
        l0_points = xyz
        l0_xyz = xyz
        # if self.normal_channel:
        #     l0_points = xyz
        #     l0_xyz = xyz[:, :3, :]
        # else:
        #     l0_points = xyz
        #     l0_xyz = xyz

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
        x = x.permute(0, 2, 1)
        return x


class PointNetSetAbstractionMsg1(nn.Module):
    def __init__(self, npoint: int, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg1, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks_0_0 = nn.Conv2d(in_channel + 3, mlp_list[0][0], 1)
        self.bn_blocks_0_0 = nn.BatchNorm2d(mlp_list[0][0])
        self.conv_blocks_0_1 = nn.Conv2d(mlp_list[0][0], mlp_list[0][1], 1)
        self.bn_blocks_0_1 = nn.BatchNorm2d(mlp_list[0][1])
        self.conv_blocks_0_2 = nn.Conv2d(mlp_list[0][1], mlp_list[0][2], 1)
        self.bn_blocks_0_2 = nn.BatchNorm2d(mlp_list[0][2])
        self.radius_0 = radius_list[0]
        self.nsample_0 = nsample_list[0]

        self.conv_blocks_1_0 = nn.Conv2d(in_channel + 3, mlp_list[1][0], 1)
        self.bn_blocks_1_0 = nn.BatchNorm2d(mlp_list[1][0])
        self.conv_blocks_1_1 = nn.Conv2d(mlp_list[1][0], mlp_list[1][1], 1)
        self.bn_blocks_1_1 = nn.BatchNorm2d(mlp_list[1][1])
        self.conv_blocks_1_2 = nn.Conv2d(mlp_list[1][1], mlp_list[1][2], 1)
        self.bn_blocks_1_2 = nn.BatchNorm2d(mlp_list[1][2])
        self.radius_1 = radius_list[1]
        self.nsample_1 = nsample_list[1]

        self.conv_blocks_2_0 = nn.Conv2d(in_channel + 3, mlp_list[2][0], 1)
        self.bn_blocks_2_0 = nn.BatchNorm2d(mlp_list[2][0])
        self.conv_blocks_2_1 = nn.Conv2d(mlp_list[2][0], mlp_list[2][1], 1)
        self.bn_blocks_2_1 = nn.BatchNorm2d(mlp_list[2][1])
        self.conv_blocks_2_2 = nn.Conv2d(mlp_list[2][1], mlp_list[2][2], 1)
        self.bn_blocks_2_2 = nn.BatchNorm2d(mlp_list[2][2])
        self.radius_2 = radius_list[2]
        self.nsample_2 = nsample_list[2]

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))    # 得到最新采样的点
        new_points_list = []

        group_idx = query_ball_point(self.radius_0, self.nsample_0, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_0_0(self.conv_blocks_0_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_1(self.conv_blocks_0_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_2(self.conv_blocks_0_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        group_idx = query_ball_point(self.radius_1, self.nsample_1, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_1_0(self.conv_blocks_1_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_1(self.conv_blocks_1_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_2(self.conv_blocks_1_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        group_idx = query_ball_point(self.radius_2, self.nsample_2, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_2_0(self.conv_blocks_2_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_2_1(self.conv_blocks_2_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_2_2(self.conv_blocks_2_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstractionMsg2(nn.Module):
    def __init__(self, npoint: int, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg2, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks_0_0 = nn.Conv2d(in_channel + 3, mlp_list[0][0], 1)
        self.bn_blocks_0_0 = nn.BatchNorm2d(mlp_list[0][0])
        self.conv_blocks_0_1 = nn.Conv2d(mlp_list[0][0], mlp_list[0][1], 1)
        self.bn_blocks_0_1 = nn.BatchNorm2d(mlp_list[0][1])
        self.conv_blocks_0_2 = nn.Conv2d(mlp_list[0][1], mlp_list[0][2], 1)
        self.bn_blocks_0_2 = nn.BatchNorm2d(mlp_list[0][2])
        self.radius_0 = radius_list[0]
        self.nsample_0 = nsample_list[0]

        self.conv_blocks_1_0 = nn.Conv2d(in_channel + 3, mlp_list[1][0], 1)
        self.bn_blocks_1_0 = nn.BatchNorm2d(mlp_list[1][0])
        self.conv_blocks_1_1 = nn.Conv2d(mlp_list[1][0], mlp_list[1][1], 1)
        self.bn_blocks_1_1 = nn.BatchNorm2d(mlp_list[1][1])
        self.conv_blocks_1_2 = nn.Conv2d(mlp_list[1][1], mlp_list[1][2], 1)
        self.bn_blocks_1_2 = nn.BatchNorm2d(mlp_list[1][2])
        self.radius_1 = radius_list[1]
        self.nsample_1 = nsample_list[1]

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))    # 得到最新采样的点
        new_points_list = []

        group_idx = query_ball_point(self.radius_0, self.nsample_0, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_0_0(self.conv_blocks_0_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_1(self.conv_blocks_0_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_2(self.conv_blocks_0_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        group_idx = query_ball_point(self.radius_1, self.nsample_1, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_1_0(self.conv_blocks_1_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_1(self.conv_blocks_1_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_2(self.conv_blocks_1_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstraction1(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel, mlp, group_all):
        super(PointNetSetAbstraction1, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs_0 = nn.Conv2d(in_channel, mlp[0], 1)
        self.mlp_bns_0 = nn.BatchNorm2d(mlp[0])
        self.mlp_convs_1 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.mlp_bns_1 = nn.BatchNorm2d(mlp[1])
        self.mlp_convs_2 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.mlp_bns_2 = nn.BatchNorm2d(mlp[2])
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_xyz, new_points = sample_and_group_all(xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)   # [B, C+D, nsample, npoint]
        new_points = F.relu(self.mlp_bns_0(self.mlp_convs_0(new_points)))
        new_points = F.relu(self.mlp_bns_1(self.mlp_convs_1(new_points)))
        new_points = F.relu(self.mlp_bns_2(self.mlp_convs_2(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation0(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation0, self).__init__()
        self.mlp_convs_0 = nn.Conv1d(in_channel, mlp[0], 1)
        self.mlp_bns_0 = nn.BatchNorm1d(mlp[0])
        self.mlp_convs_1 = nn.Conv1d(mlp[0], mlp[1], 1)
        self.mlp_bns_1 = nn.BatchNorm1d(mlp[1])

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        # print("type S: ", type(S), S, isinstance(S, int), torch.is_tensor(S))
        interpolated_points = points2.repeat(1, N, 1)

        points1 = points1.permute(0, 2, 1)
        new_points = torch.cat([points1, interpolated_points], dim=-1)

        new_points = new_points.permute(0, 2, 1)
        # for conv, bn in zip(self.mlp_convs, self.mlp_bns):
        #     new_points = F.relu(bn(conv(new_points)))
        new_points = F.relu(self.mlp_bns_0(self.mlp_convs_0(new_points)))
        new_points = F.relu(self.mlp_bns_1(self.mlp_convs_1(new_points)))
        return new_points


class PointNetFeaturePropagation1(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation1, self).__init__()
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # last_channel = in_channel
        # for out_channel in mlp:
        #     self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm1d(out_channel))
        #     last_channel = out_channel
        self.mlp_convs_0 = nn.Conv1d(in_channel, mlp[0], 1)
        self.mlp_bns_0 = nn.BatchNorm1d(mlp[0])
        self.mlp_convs_1 = nn.Conv1d(mlp[0], mlp[1], 1)
        self.mlp_bns_1 = nn.BatchNorm1d(mlp[1])

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        # print("type S: ", type(S), S, isinstance(S, int), torch.is_tensor(S))
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        points1 = points1.permute(0, 2, 1)
        new_points = torch.cat([points1, interpolated_points], dim=-1)

        new_points = new_points.permute(0, 2, 1)
        # for conv, bn in zip(self.mlp_convs, self.mlp_bns):
        #     new_points = F.relu(bn(conv(new_points)))
        new_points = F.relu(self.mlp_bns_0(self.mlp_convs_0(new_points)))
        new_points = F.relu(self.mlp_bns_1(self.mlp_convs_1(new_points)))
        return new_points


class GetModel1(nn.Module):
    def __init__(self, num_classes, normal_channel=False, num_categories=16):
        super(GetModel1, self).__init__()
        self.num_categories = num_categories
        additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg1(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg2(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction1(npoint=256, radius=5.0, nsample=256, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation0(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation1(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation1(in_channel=134 + self.num_categories + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
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
        x = x.permute(0, 2, 1)
        return x


if __name__ == "__main__":
    # # model
    # net = GetModel(2, normal_channel=False, num_categories=1)
    # checkpoint = torch.load("../log/part_seg/pointnet2_part_seg_msg_add_data/checkpoints/best_model.pth", map_location=torch.device('cpu'))
    # net.load_state_dict(checkpoint['model_state_dict'])
    # net = net.eval()

    net = GetModel1(2, normal_channel=False, num_categories=1)
    state_dict = torch.load("../log/part_seg/pointnet2_part_seg_msg_add_data/checkpoints/best_model_1.3.0_new.pth", map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    net = net.eval()

    cate = torch.ones((1, 1, 1))
    points = torch.randn((1, 3, 6000))
    out1 = net(points, cate)
    print(points.shape, cate.shape)
    print("out shape: ", out1.shape)
    print("net out: ", out1)

    onnx_path = "./model_ori.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,   # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (points, cate),
                      onnx_path,
                      verbose=True,
                      input_names=["points", "cate"],
                      output_names=["cls_prob"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                          # dict value: manually named axes
                          "points": {0: "batch_size", 1: "channel", 2: "n_points"},
                          # list value: automatic names
                          "cls_prob": {0: "batch_size", 1: "n_points", 2: "n_cls"},
                      }
                      )

    print("onnx model has exported!")

    # inference by onnx
    import onnxruntime
    import onnx
    import os
    import onnxoptimizer

    # check
    print("start check model ...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    print("model is OK!")

    # # # optimizer model
    # # print("start optimizer model ....")
    # # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    # # optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    # # onnx.save(optimized_model, "./model_ori_optimizer.onnx")
    # # print("model was optimized, and saved in ", "./model_ori_optimizer.onnx")
    # # shared_library = "./inference_onnx/fps.dll"

    so1 = onnxruntime.SessionOptions()
    # so1.register_custom_ops_library(shared_library)
    available_providers = onnxruntime.get_available_providers()
    import time
    start = time.time()
    net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    end = time.time()
    print("load model time: ", end - start)
    start = time.time()
    out = net_session.run(None, {"points": points.numpy(), "cate": cate.numpy()})
    end = time.time()
    print("predict time: ", end - start)
    print(out)
    print(out[0].shape)
