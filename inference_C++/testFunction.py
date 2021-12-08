import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


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
    B = int(B)
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
    # B = int(B)
    # N = int(N)
    # C = int(C)

    centroids = torch.zeros(B, int(npoint), dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)  # torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random choice

    _, farthest = xyz[:, :, 1].max(1)  # torch.max(xyz[:, :, 1], 1)[1]

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(int(npoint)):
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
    def __init__(self):
        super(TestNet, self).__init__()

    def forward(self, xyz):
        # xyz==> (B, N, C)
        xyz = xyz.permute(0, 1, 3, 2)
        xyz = xyz.squeeze(0)
        select = torch.argmax(xyz, 1)
        # npoint = 2
        # select = farthest_point_sample(xyz, npoint)
        return select


class ArgMaxLayer(object):
    def __init__(self, params, blobs):
        print("params: ", params)
        self.axis = params["axis"]

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        print("memory shape", inputs)
        out_dim = []
        input_dim = inputs[0]
        for i in range(len(input_dim)):
            if i != self.axis:
                out_dim.append(input_dim[i])
        print(out_dim)
        return [out_dim]

    def forward(self, inputs):
        print("inputs-: ", inputs)
        # TODO: find max ids on axis
        data = inputs[self.axis]
        print("data: ", data)
        return np.array([[0, 2, 4]])


cv2.dnn_registerLayer('ArgMax', ArgMaxLayer)


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

    net = TestNet()
    inputs = torch.randn((1, 3, 25))
    inputs = inputs.unsqueeze(0)
    out = net(inputs)
    print("**** torch out ******: ", out)
    onnx_path = "./test.onnx"
    print("start convert model to onnx >>>")
    torch.onnx.export(net,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (inputs, ),
                      onnx_path,
                      verbose=True,
                      input_names=["points"],
                      output_names=["res"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                          "points": {1: "b", 2: "c", 3: "n"},
                          "res": {0: "b", 1: "n", 2: "c"}
                      }
                      )

    print("onnx model has exported!")

    # inference by onnx
    import onnxruntime
    import onnx
    # check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    so1 = onnxruntime.SessionOptions()
    available_providers = onnxruntime.get_available_providers()

    net_session = onnxruntime.InferenceSession(onnx_path, sess_options=so1, providers=available_providers)
    out = net_session.run(None, {"points": inputs.numpy()})
    print("----onnx runtime out----: ", out)

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
    net.setInput(inputs.numpy().astype(np.float32), name="points")
    print("set input Done")
    print("$$$$$cv res$$$$: ", net.forward())