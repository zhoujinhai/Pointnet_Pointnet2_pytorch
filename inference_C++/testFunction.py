import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def process_data(input_data, is_file=True):
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

    point_set = data[:, 0:3]

    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    choice = np.random.choice(range(0, len(point_set)), len(point_set), replace=False)
    point_set = point_set[choice, :]

    point_set = np.expand_dims(point_set, axis=0)
    return torch.from_numpy(point_set), choice


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
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # print("sqrdists: ", sqrdists, sqrdists.shape)
    # mask = sqrdists > radius ** 2
    # group_idx[mask] = N
    temp = torch.full((B, S, N), N, dtype=torch.long)  # torch.ones(group_idx.shape, dtype=torch.long) * N
    group_idx = torch.where(sqrdists > radius ** 2, temp, group_idx)

    group_idx = group_idx.sort()[0]
    # print("group_idx: ", group_idx, group_idx.shape)
    group_idx = group_idx[:, :, :nsample]
    # print("group_idx slice: ", group_idx, group_idx.shape)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    # print("N: ", N)
    group_idx = torch.where(group_idx == N, group_first, group_idx)
    return group_idx


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
    # B = int(B)
    view_shape = list(idx.shape)
    # print("view_shape: ", view_shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    for i in range(1, len(view_shape)):
        view_shape[i] = 1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # print("view_shape: ", view_shape, " repeat_shape: ", repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    # print("batch_indices: ", batch_indices, batch_indices.shape)
    new_points = points[batch_indices, idx, :]
    # print("new_points: ", new_points, new_points.shape)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, int(npoint), dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)  # torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random choice
    # print(torch.max(xyz, 1)[1][:, 1].type(torch.long))
    farthest = torch.max(xyz[:, :, 1], 1)[1].to(torch.long)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(int(npoint)):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # mask = dist < distance
        # distance[mask] = dist[mask]
        distance = torch.where(dist < distance, dist, distance)
        farthest = torch.max(distance, 1)[1]

    return centroids


def sub_center(grouped_xyz, new_xyz):
    """
    pytorch broadcast operator
    :param grouped_xyz:  B * S * N * C
    :param new_xyz: B * S * C
    :param B:
    :param S:
    :param C:
    :return: B * S * N * C
    """
    B, S, C = new_xyz.shape
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    return grouped_xyz


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
    B, N, D = points.shape
    # B, N, C = int(B), int(N), int(C)    # this dim is fixed
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    new_points = torch.cat([grouped_xyz, points.view(B, 1, N, D)], dim=-1)
    # if points is not None:
    #     new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    # else:
    #     new_points = grouped_xyz
    return new_xyz, new_points


def get_cate(xyz, n):
    B, C, N = xyz.shape
    res = torch.ones((1, 1, 1)).view(B, n, 1).repeat(1, 1, N)
    return res


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
        # B, N, C = int(B), int(N), int(C)
        S = self.npoint
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S))    # 得到最新采样的点
        # print("new_xyz: ", new_xyz.shape)
        new_xyz = torch.ops.my_ops.idx_pts(xyz, torch.ops.my_ops.fps(xyz, S), xyz.shape[-1]).squeeze(0)

        new_points_list = []

        # group_idx = query_ball_point(self.radius_0, self.nsample_0, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_0), self.nsample_0, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx, xyz.shape[-1]).squeeze(0)
        # grouped_xyz = sub_center(grouped_xyz, new_xyz, B, S, C)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz)  # , B, S, C
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx, points.shape[-1]).squeeze(0)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_0_0(self.conv_blocks_0_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_1(self.conv_blocks_0_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_2(self.conv_blocks_0_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        # group_idx = query_ball_point(self.radius_1, self.nsample_1, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_1), self.nsample_1, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx, xyz.shape[-1]).squeeze(0)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx, points.shape[-1]).squeeze(0)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_1_0(self.conv_blocks_1_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_1(self.conv_blocks_1_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_1_2(self.conv_blocks_1_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        # group_idx = query_ball_point(self.radius_2, self.nsample_2, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_2), self.nsample_2, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx, xyz.shape[-1]).squeeze(0)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx, points.shape[-1]).squeeze(0)
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
        # B, N, C = int(B), int(N), int(C)
        S = self.npoint  # int(self.npoint)
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S))    # 得到最新采样的点
        new_xyz = torch.ops.my_ops.idx_pts(xyz, torch.ops.my_ops.fps(xyz, S), xyz.shape[-1]).squeeze(0)
        new_points_list = []

        # group_idx = query_ball_point(self.radius_0, self.nsample_0, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_0), self.nsample_0, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx, xyz.shape[-1]).squeeze(0)
        # grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx, points.shape[-1]).squeeze(0)   # last dim is 320 not 3
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        grouped_points = F.relu(self.bn_blocks_0_0(self.conv_blocks_0_0(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_1(self.conv_blocks_0_1(grouped_points)))
        grouped_points = F.relu(self.bn_blocks_0_2(self.conv_blocks_0_2(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        new_points_list.append(new_points)

        # group_idx = query_ball_point(self.radius_1, self.nsample_1, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_1), self.nsample_1, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx, xyz.shape[-1]).squeeze(0)
        # grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx, points.shape[-1]).squeeze(0)  # last dim is 320 not 3
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
        # print("********new_xyz shape: ", new_xyz.shape)
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
        # B, N, C, S = int(B), int(N), int(C), int(S)
        # print("type S: ", type(S), S, isinstance(S, int), torch.is_tensor(S))
        interpolated_points = points2.repeat(1, N, 1)  # interpolated_points = points2.expand(-1, N, -1)

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
        # dists = torch.ops.my_ops.dist(xyz1, xyz2, xyz1.shape[1]).squeeze(0)  # square_distance(xyz1, xyz2)
        # # print("dists shape: ", xyz1.shape, xyz2.shape, dists.shape)
        #
        # dists, idx = dists.sort(dim=2)
        # dists = dists[:, :, :3]
        # idx = idx[:, :, :3]  # [B, N, 3]

        # dists = torch.ops.my_ops.dist(xyz1, xyz2)  # , xyz1.shape[1]
        # dists, idx = dists.sort(dim=3)   # for cv not support 3D output
        # dists = dists.squeeze(0)
        # idx = idx.squeeze(0)
        # dists = dists[:, :, :3]
        # idx = idx[:, :, :3]  # [B, N, 3]
        #
        # dist_recip = 1.0 / (dists + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight = dist_recip / norm
        # # interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        # interpolated_points = torch.sum(torch.ops.my_ops.idx_pts(points2, idx, points2.shape[-1]).squeeze(0) * weight.view(B, N, 3, 1), dim=2)
        # print("interpolated_points: ", interpolated_points.shape)
        # points1 = points1.permute(0, 2, 1)
        # new_points = torch.cat([points1, interpolated_points], dim=-1)
        # print("new_points: ", new_points.shape)
        # print("xyz1, xyz2, points1, points2: ", xyz1.shape, xyz2.shape, points1.shape, points2.shape)
        new_points = torch.ops.my_ops.propagatedata(xyz1, xyz2, points1, points2).squeeze(0)

        new_points = new_points.permute(0, 2, 1)
        # for conv, bn in zip(self.mlp_convs, self.mlp_bns):
        #     new_points = F.relu(bn(conv(new_points)))
        new_points = F.relu(self.mlp_bns_0(self.mlp_convs_0(new_points)))
        new_points = F.relu(self.mlp_bns_1(self.mlp_convs_1(new_points)))
        return new_points


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, xyz):
        npoint = 2
        # cls_label = torch.ones((1, 1, 1))
        xyz = xyz.permute(0, 1, 3, 2)
        xyz = xyz.squeeze(0)

        # B, N, C = xyz.shape
        # select = torch.argmax(xyz, 1)
        # print(xyz.shape)
        # print("select: ", select, select.shape)
        select = farthest_point_sample(xyz, npoint)
        # new_xyz = index_points(xyz, select)
        # idx = query_ball_point(0.5, 1, xyz, new_xyz)
        return select  # idx


class TestNet(nn.Module):
    def __init__(self, num_classes=2, normal_channel=False, num_categories=1):
        super(TestNet, self).__init__()
        self.num_categories = num_categories
        self.sa1 = PointNetSetAbstractionMsg1(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg2(256, [0.4, 0.8], [64, 128], 128 + 128 + 64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction1(npoint=256, radius=5.0, nsample=256, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation0(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation1(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation1(in_channel=134 + self.num_categories, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # # xyz==> (B, N, C)
        # xyz = xyz.permute(0, 1, 3, 2)
        # xyz = xyz.squeeze(0)
        #
        # npoint = 2
        # fps_idx = torch.ops.my_ops.fps(xyz, npoint)
        # new_xyz = torch.ops.my_ops.idx_pts(xyz, fps_idx).squeeze(0)  # 得到最新采样的点
        # # print(xyz.shape, new_xyz.shape)
        # group_idx = torch.ops.my_ops.query_ball_pts(self.radius_0, self.nsample_0, xyz, new_xyz).squeeze(0)
        # # print("group_idx: ", group_idx.shape)
        # grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx).squeeze(0)
        # # print("grouped_xyz: ", grouped_xyz.shape)
        # return grouped_xyz

        xyz = xyz.squeeze(0)    # xyz is B * C * N
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # 1 * 3 * 256   1 * 512 * 256
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # # cls_label_one_hot = get_cate(xyz, self.num_categories)
        # cls_label_one_hot = torch.ones(B, self.num_categories, N)  # torch.ones((1, 1, 1)).view(B, self.num_categories, 1).repeat(1, 1, N)
        # print("**** cls_label_one_hot: ", cls_label_one_hot.shape)
        cat_data_cate = torch.ops.my_ops.get_cate(l0_xyz, l0_points, self.num_categories, C).squeeze(0)
        # print("cat_data_cate: ", cat_data_cate.shape)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, cat_data_cate, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x


# class TestNet(nn.Module):
#     def __init__(self):
#         super(TestNet, self).__init__()
#
#     def forward(self, xyz, center):
#         """
#         :param xyz: B * N * S * C == 1 * 2 * 4 * 3
#         :param center: B * N * C == 1 * 2 * 3
#         :return:
#         """
#         res = xyz - center.view(1, 2, 1, 3)   # .repeat([1, 1, 4, 1])
#
#         return res


class ArgMaxLayer(object):
    def __init__(self, params, blobs):
        print("params: ", params)
        self.axis = params["axis"]
        self.dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("memory shape", inputs)
        out_dim = []
        input_dim = inputs[0]
        for i in range(len(input_dim)):
            if i != self.axis:
                out_dim.append(input_dim[i])
        print("out_dim", out_dim)
        self.dim = out_dim
        return [out_dim]

    def forward(self, inputs):
        data = inputs[0]
        print("inputs-: ", type(data), data.dtype, data.shape)
        # find max ids on axis
        res = np.argmax(data, axis=self.axis).astype(np.float32)
        # print("axis: ", self.axis)
        # shape = data.shape
        # print("shape: ", shape)
        # res = np.random.randint(0, shape[self.axis], tuple(self.dim), dtype=np.longlong)
        print(res, res.shape, res.dtype)
        return [res]


class FPSLayer(object):
    def __init__(self, params, blobs):
        # print("params: ", params)
        # print("blobs: ", blobs)
        self.npoint = int(blobs[0][0][0])

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        # print("memory shape inputs: ", inputs)
        out_dim = []
        input_dim = inputs[0]
        out_dim.append(input_dim[0])
        out_dim.append(self.npoint)
        print("fps out_dim: ", out_dim)
        return [out_dim]

    def forward(self, inputs):
        # print("inputs size: ", len(inputs))
        data = inputs[0]
        # print("data -: ", type(data), data.dtype, data.shape)
        # TODO: get the farthest point's ids
        rand_data = np.random.randint(0, data.shape[1], (data.shape[0], self.npoint))
        res = rand_data.astype(np.float32)
        print("fps res: ", res.shape, res.dtype)
        return [res]


class IndexPtsLayer(object):
    def __init__(self, params, blobs):
        self.out_dim = None
        print("index params: ", params)
        print("index blobs: ", blobs)
        self.channel = int(blobs[0][0][0])

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        out_dim = []
        input_dim, idx_dim = inputs
        print("IndexPtsLayer memory: ", inputs)
        out_dim.append(1)   # python may not support 3D output
        out_dim.append(input_dim[0])
        for i in range(1, len(idx_dim)):
            out_dim.append(idx_dim[i])
        out_dim.append(self.channel)   # input_dim[2]   # 直接设为3遇到其他维度会有问题
        print("IndexPtsLayer out_dim: ", out_dim)
        # out_dim[-1] = 3
        # print("out:***********", out_dim)
        self.out_dim = out_dim
        return [out_dim]

    def forward(self, inputs):
        print("IndexPtsLayer inputs size: ", len(inputs))
        data, idx = inputs
        # print("idx: ", idx)
        print("IndexPtsLayer inputs : ", data.shape, idx.shape, self.out_dim)
        # TODO: get the point's with idx
        rand_data = np.random.randint(1, data.shape[1], self.out_dim).astype(np.float32)
        print("IndexPtsLayer res: ", rand_data.shape)

        return [rand_data]


class QueryBallPtsLayer(object):
    def __init__(self, params, blobs):
        # print("params: ", params)
        # print("blobs: ", blobs)
        self.radius = float(blobs[0][0][0])
        self.nsample = int(blobs[1][0][0])
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        out_dim = []

        xyz_dim, new_xyz_dim = inputs
        out_dim.append(1)   # python not support 3D output
        out_dim.append(xyz_dim[0])      # batch
        out_dim.append(new_xyz_dim[1])  # new_xyz nsample
        out_dim.append(self.nsample)
        print("QueryBallPtsLayer out_dim: ", out_dim)
        self.out_dim = out_dim
        return [out_dim]

    def forward(self, inputs):
        print("QueryBallPtsLayer inputs size: ", len(inputs))
        data, idx = inputs
        # print("idx: ", idx)
        print("QueryBallPtsLayer data : ", data.shape, idx.shape, self.out_dim)
        # TODO: get the idx with self.radius and self.nsample
        rand_data = np.random.randint(1, data.shape[1], self.out_dim).astype(np.float32)
        print("QueryBallPtsLayer res: ", rand_data.shape)

        return [rand_data]


class SubCenterLayer(object):
    def __init__(self, params, blobs):
        print("SubCenterLayer params: ", params)
        print("SubCenterLayer blobs: ", blobs)
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("SubCenterLayer inputs: ", inputs)
        group_xyz, new_xyz_dim = inputs
        out_dim = group_xyz
        self.out_dim = out_dim
        print("SubCenterLayer out_dim: ", self.out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("SubCenterLayer inputs size: ", len(inputs))
        grouped_xyz, new_xyz = inputs
        # print("idx: ", idx)
        print("SubCenterLayer data : ", grouped_xyz.shape, new_xyz.shape, self.out_dim)
        new_xyz_shape = list(new_xyz.shape)
        new_xyz_shape.insert(2, 1)
        new_xyz = new_xyz.reshape(new_xyz_shape)
        res = grouped_xyz - new_xyz
        print("SubCenterLayer res: ", res.shape)

        return [res]


class TileLayer(object):
    def __init__(self, params, blobs):
        print("TileLayer params: ", params)
        print("TileLayer blobs: ", blobs)
        # print("B, S, C", self.B, self.S, self.C)
        self.out_dim = None

        blob = blobs[0]

        for i in range(len(blob)):
            if blob[i][0] != 1:
                self.repeat = int(blob[i][0])
                self.idx = int(i)
        print("------------self.repeat: , self.idx", self.repeat, self.idx)

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("TileLayer inputs: ", inputs)
        input_dim = inputs[0]
        out_dim = input_dim
        out_dim[self.idx] = self.repeat
        self.out_dim = out_dim
        print("TileLayer out_dim: ", self.out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("TileLayer inputs size: ", len(inputs))
        data = inputs[0]
        # print("idx: ", idx)
        print("TileLayer data : ", data.shape, self.out_dim)
        # TODO implement torch.repeat
        res = np.random.randn(self.out_dim[0], self.out_dim[1], self.out_dim[2]).astype(np.float32)
        print("TileLayer res: ", res.shape)

        return [res]


class TopKLayer(object):
    def __init__(self, params, blobs):
        print("TopKLayer params: ", params)
        print("TopKLayer blobs: ", blobs)
        # print("B, S, C", self.B, self.S, self.C)
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("TopKLayer inputs: ", inputs)
        input_dim = inputs[0]
        out_dim = input_dim

        self.out_dim = out_dim
        print("TopKLayer out_dim: ", self.out_dim)
        return [out_dim, out_dim]

    def forward(self, inputs):
        print("TopKLayer inputs size: ", len(inputs))
        data = inputs[0]
        # print("idx: ", idx)
        print("TopKLayer data : ", data.shape, self.out_dim)
        idx = data.argsort(axis=3).astype(np.float32)
        data.sort(axis=3)
        print("TopKLayer res: ", data.shape, idx.shape)

        return [data, idx]


class DistLayer(object):
    def __init__(self, params, blobs):
        print("DistLayer params: ", params)
        print("DistLayer blobs: ", blobs)
        # print("B, S, C", self.B, self.S, self.C)
        # self.N = int(blobs[0][0][0])
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("DistLayer inputs: ", inputs)
        src_dim, dst_dim = inputs
        out_dim = []
        out_dim.append(1)
        out_dim.append(src_dim[0])
        out_dim.append(src_dim[1])
        out_dim.append(dst_dim[1])

        self.out_dim = out_dim
        print("DistLayer out_dim: ", self.out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("DistLayer inputs size: ", len(inputs))
        res = np.ones(self.out_dim).astype(np.float32)

        print("DistLayer res: ", res.shape, res.shape)

        return [res]


class GetCateLayer(object):
    def __init__(self, params, blobs):
        print("GetCateLayer params: ", params)
        print("GetCateLayer blobs: ", blobs)
        # print("B, S, C", self.B, self.S, self.C)
        n_cls = int(blobs[0][0][0])
        C = int(blobs[1][0][0])
        self.C = n_cls + 2 * C
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("GetCateLayer inputs: ", inputs)
        src_dim, dst_dim = inputs
        out_dim = []
        out_dim.append(1)
        out_dim.append(src_dim[0])
        out_dim.append(self.C)
        out_dim.append(src_dim[2])

        self.out_dim = out_dim
        print("GetCateLayer out_dim: ", self.out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("GetCateLayer inputs size: ", len(inputs))
        # TODO implement concat (cate, xyz, points)
        res = np.ones(self.out_dim).astype(np.float32)

        print("GetCateLayer res: ", res.shape, res.shape)

        return [res]


class PropagateDataLayer(object):
    def __init__(self, params, blobs):
        print("PropagateDataLayer params: ", params)
        print("PropagateDataLayer blobs: ", blobs)
        # print("B, S, C", self.B, self.S, self.C)
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("PropagateDataLayer inputs: ", inputs)
        xyz1, xyz2, points1, points2 = inputs
        out_channel = int(points1[1] + points2[2])
        out_dim = []
        out_dim.append(1)
        out_dim.append(xyz1[0])
        out_dim.append(xyz1[1])
        out_dim.append(out_channel)
        self.out_dim = out_dim
        print("PropagateDataLayer out_dim: ", self.out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("PropagateDataLayer inputs size: ", len(inputs))
        # TODO implement concat (cate, xyz, points)
        res = np.ones(self.out_dim).astype(np.float32)

        print("PropagateDataLayer res: ", res.shape, res.shape)

        return [res]


# cv2.dnn_registerLayer('ArgMax', ArgMaxLayer)
cv2.dnn_registerLayer('fps', FPSLayer)
cv2.dnn_registerLayer('idx_pts', IndexPtsLayer)
cv2.dnn_registerLayer('query_ball_pts', QueryBallPtsLayer)
cv2.dnn_registerLayer('sub_center', SubCenterLayer)
cv2.dnn_registerLayer('Tile', TileLayer)
# cv2.dnn_registerLayer('TopK', TopKLayer)
# cv2.dnn_registerLayer('dist', DistLayer)
cv2.dnn_registerLayer('get_cate', GetCateLayer)
cv2.dnn_registerLayer("propagatedata", PropagateDataLayer)


if __name__ == "__main__":
    # net = Net()
    # inputs = torch.randn((1, 3, 35))
    # out = net(inputs.unsqueeze(0))
    # print("**** torch out ******: ", out)
    # onnx_path = "./test.onnx"
    # # print("start convert model to onnx >>>")
    # # torch.onnx.export(net,   # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
    # #                   (inputs.unsqueeze(0), ),
    # #                   onnx_path,
    # #                   verbose=True,
    # #                   input_names=["points"],
    # #                   output_names=["select_ids"],
    # #                   opset_version=12,
    # #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
    # #                   dynamic_axes={
    # #                      "points": {1: "b", 2: "c", 3: "n"},
    # #                      "select_ids": {0: "b", 1: "n", 2: "c"}
    # #                   }
    # #                   )
    # #
    # # print("onnx model has exported!")
    #
    # # inference by onnx
    # import onnxruntime
    # import onnx
    # import os
    #
    # # check
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    # with open("./temp1.txt", "w") as f:
    #     f.write(onnx.helper.printable_graph(onnx_model.graph))
    #
    # so1 = onnxruntime.SessionOptions()
    # available_providers = onnxruntime.get_available_providers()
    #
    # net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    # out = net_session.run(None, {"points": inputs.unsqueeze(0).numpy()})
    # print("onnx runtime out: ", out)
    #
    # import cv2
    # net = cv2.dnn.readNetFromONNX(onnx_path)
    # # 获取各层信息
    # layer_names = net.getLayerNames()
    #
    # for name in layer_names:
    #     id = net.getLayerId(name)
    #     layer = net.getLayer(id)
    #     print("layer id : %d, type : %s, name: %s" % (id, layer.type, layer.name))
    #
    # print("cv2 load model is OK!")
    # print("start set input")
    # net.setInput(inputs.unsqueeze(0).numpy(), name="points")
    # print("set input 1 Done")
    #
    # print("res: ", net.forward())

    # data_path = r"D:\Debug_dir\news_data\pcd_label_normal\bankou (1)_minCruv.pcd"
    # points, choices = process_data(data_path)
    # points = points.transpose(2, 1)
    # print(points.shape)
    # inputs = points

    torch.ops.load_library("./inference_onnx/fps.dll")
    torch.ops.load_library("./inference_onnx/idx_pts.dll")
    torch.ops.load_library("./inference_onnx/query_ball_pts.dll")
    torch.ops.load_library("./inference_onnx/sub_center.dll")
    # torch.ops.load_library("./inference_onnx/sample_group_all.dll")
    torch.ops.load_library("./inference_onnx/get_cate.dll")
    # torch.ops.load_library("./inference_onnx/dist.dll")
    torch.ops.load_library("./inference_onnx/propagatedata.dll")


    def my_fps(g, xyz, npoints):
        return g.op("my_ops::fps", xyz, npoints)

    def my_idx_pts(g, xyz, idx, channel):
        return g.op("my_ops::idx_pts", xyz, idx, channel)

    def my_query_ball_pts(g, radius, nsample, xyz, new_xyz):
        return g.op("my_ops::query_ball_pts", radius, nsample, xyz, new_xyz)

    def my_sub_center(g, grouped_xyz, new_xyz):
        return g.op("my_ops::sub_center", grouped_xyz, new_xyz)

    # def my_sample_and_group_all(g, xyz, points):
    #     return g.op("my_ops::sample_and_group_all", xyz, points)

    def my_get_cate(g, xyz, points, n_c, C):
        return g.op("my_ops::get_cate", xyz, points, n_c, C)

    # def my_dist(g, src, dst):
    #     return g.op("my_ops::dist", src, dst)

    def my_propagatedata(g, xyz1, xyz2, points1, points2):
        return g.op("my_ops::propagatedata", xyz1, xyz2, points1, points2)

    torch.onnx.register_custom_op_symbolic("my_ops::fps", my_fps, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::idx_pts", my_idx_pts, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::query_ball_pts", my_query_ball_pts, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::sub_center", my_sub_center, 9)
    # torch.onnx.register_custom_op_symbolic("my_ops::sample_and_group_all", my_sample_and_group_all, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::get_cate", my_get_cate, 9)
    # torch.onnx.register_custom_op_symbolic("my_ops::dist", my_dist, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::propagatedata", my_propagatedata, 9)

    net = TestNet()
    state_dict = torch.load("../log/part_seg/pointnet2_part_seg_msg_add_data/checkpoints/best_model_1.3.0_new.pth")
    net.load_state_dict(state_dict)
    inputs = torch.randn((1, 3, 2500))  # B, C, N
    inputs = inputs.unsqueeze(0)

    out = net(inputs)
    print("**** torch out ******: ", out.shape)
    onnx_path = "./test.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (inputs,),
                      onnx_path,
                      verbose=True,
                      input_names=["points"],
                      output_names=["res"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                          "points": {1: "b", 2: "c", 3: "n"},
                          "res": {0: "b", 1: "n", 2: "n_c"}
                      }
                      )

    print("onnx model has exported!")

    # # inference by onnx
    # import onnxruntime
    # import onnx
    # # check
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    # so1 = onnxruntime.SessionOptions()
    # available_providers = onnxruntime.get_available_providers()
    #
    # net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    # out = net_session.run(None, {"points": inputs.numpy(), "center": center.numpy()})
    # print("----onnx runtime out----: ", out)

    import cv2
    net = cv2.dnn.readNetFromONNX(onnx_path)
    # 获取各层信息
    layer_names = net.getLayerNames()

    for name in layer_names:
        id = net.getLayerId(name)
        layer = net.getLayer(id)
        print("layer id : %d, type : %s, name: %s" % (id, layer.type, layer.name))

    print("cv2 load model is OK!")
    print("start set input")
    print("inputs shape: ", inputs.shape)
    net.setInput(inputs.numpy().astype(np.float32), name="points")
    # net.setInput(center.numpy().astype(np.float32), name="center")
    print("set input Done")
    cv_res = net.forward()
    print("$$$$$cv res$$$$: ", cv_res.shape, cv_res.dtype, type(cv_res))
