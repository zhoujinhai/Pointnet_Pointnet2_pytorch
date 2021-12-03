import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # mask = sqrdists > radius ** 2
    # group_idx[mask] = N
    temp = torch.full((B, S, N), N, dtype=torch.long)  # torch.ones(group_idx.shape, dtype=torch.long) * N
    group_idx = torch.where(sqrdists > radius ** 2, temp, group_idx)

    group_idx = group_idx.sort()[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
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
    view_shape = list(idx.shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    for i in range(1, len(view_shape)):
        view_shape[i] = 1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    print("view_shape: ", view_shape, " repeat_shape: ", repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)  # torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random choice
    _, farthest = xyz[:, :, 0].max(1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # mask = dist < distance
        # distance[mask] = dist[mask]
        distance = torch.where(dist < distance, dist, distance)
        farthest = torch.max(distance, -1)[1]

    return centroids


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, xyz, cls_label, npoint: int):
        xyz = xyz.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        idx = query_ball_point(0.5, 1, xyz, new_xyz)
        return idx


if __name__ == "__main__":
    net = Net()
    inputs = torch.randn((1, 3, 20))
    cate = torch.ones((1, 1, 1))
    n_pts = torch.tensor(5)
    out = net(inputs, cate, n_pts)
    print(out, out.shape)
    onnx_path = "./test.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,   # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (inputs, cate, n_pts),
                      onnx_path,
                      verbose=True,
                      input_names=["points", "cate", "npoint"],
                      output_names=["select_ids"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                         "points": {0: "b", 1: "c", 2: "n"},
                         "select_ids": {0: "b", 1: "n1", 3: "c"}
                      }
                      )

    print("onnx model has exported!")

    # inference by onnx
    import onnxruntime
    import onnx
    import os

    # check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    so1 = onnxruntime.SessionOptions()
    available_providers = onnxruntime.get_available_providers()

    net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    out = net_session.run(None, {"points": inputs.numpy(), "cate": cate.numpy(), "npoint": n_pts.numpy()})
    print("out: ", out)

    import cv2
    net = cv2.dnn.readNetFromONNX(onnx_path)