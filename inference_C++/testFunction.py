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
    print("sqrdists: ", sqrdists, sqrdists.shape)
    # mask = sqrdists > radius ** 2
    # group_idx[mask] = N
    temp = torch.full((B, S, N), N, dtype=torch.long)  # torch.ones(group_idx.shape, dtype=torch.long) * N
    group_idx = torch.where(sqrdists > radius ** 2, temp, group_idx)

    group_idx = group_idx.sort()[0]
    print("group_idx: ", group_idx, group_idx.shape)
    group_idx = group_idx[:, :, :nsample]
    print("group_idx slice: ", group_idx, group_idx.shape)
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
    print("view_shape: ", view_shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    for i in range(1, len(view_shape)):
        view_shape[i] = 1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    print("view_shape: ", view_shape, " repeat_shape: ", repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    print("batch_indices: ", batch_indices, batch_indices.shape)
    new_points = points[batch_indices, idx, :]
    print("new_points: ", new_points, new_points.shape)
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


def sub_center(grouped_xyz, new_xyz, B, S, C):
    """
    pytorch broadcast operator
    :param grouped_xyz:  B * S * N * C
    :param new_xyz: B * S * 3
    :param B:
    :param S:
    :param C:
    :return: B * S * N * C
    """
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    return grouped_xyz


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
        new_xyz = torch.ops.my_ops.idx_pts(xyz, torch.ops.my_ops.fps(xyz, S)).squeeze(0)

        new_points_list = []

        # group_idx = query_ball_point(self.radius_0, self.nsample_0, xyz, new_xyz)
        group_idx = torch.ops.my_ops.query_ball_pts(torch.tensor(self.radius_0), self.nsample_0, xyz, new_xyz).squeeze(0)
        # grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx).squeeze(0)
        # print("grouped_xyz: ", grouped_xyz.shape, "new_xyz shape: ", new_xyz.shape, self.nsample_0)
        # new_xyz_repeat = new_xyz.view(B, S, 1, C).repeat([1, 1, self.nsample_0, 1])
        # print("new_xyz: ", new_xyz_repeat.shape)
        # grouped_xyz = grouped_xyz - new_xyz_repeat  # new_xyz.view(B, S, 1, C)   # 减去中心点
        # print("grouped_xyz after: ", grouped_xyz.shape, "B:", B, " S: ", S, " C: ", C)
        # grouped_xyz = sub_center(grouped_xyz, new_xyz, B, S, C)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz, B, S, C)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx).squeeze(0)
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
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx).squeeze(0)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz, B, S, C)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx).squeeze(0)
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
        grouped_xyz = torch.ops.my_ops.idx_pts(xyz, group_idx).squeeze(0)
        grouped_xyz = torch.ops.my_ops.sub_center(grouped_xyz, new_xyz, B, S, C)  # grouped_xyz -= new_xyz.view(B, S, 1, C)
        # grouped_points = index_points(points, group_idx)
        grouped_points = torch.ops.my_ops.idx_pts(points, group_idx).squeeze(0)
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
        self.sa1 = PointNetSetAbstractionMsg1(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.radius_0 = torch.tensor(0.1)
        # self.nsample_0 = 32

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

        xyz = xyz.squeeze(0)
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        return l1_points


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
        print("fps res: ", res, res.shape, res.dtype)
        return [res]


class IndexPtsLayer(object):
    def __init__(self, params, blobs):
        self.out_dim = None

    # Our layer receives one inputs. We need to find the max
    def getMemoryShapes(self, inputs):
        out_dim = []
        input_dim, idx_dim = inputs
        print("IndexPtsLayer memory: ", inputs)
        out_dim.append(1)   # python may not support 3D output
        out_dim.append(input_dim[0])
        for i in range(1, len(idx_dim)):
            out_dim.append(idx_dim[i])
        out_dim.append(3)   # input_dim[2]
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
        self.B = int(blobs[0][0][0])
        self.S = int(blobs[1][0][0])
        self.C = int(blobs[2][0][0])
        # print("B, S, C", self.B, self.S, self.C)
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
        print("QueryBallPtsLayer inputs size: ", len(inputs))
        grouped_xyz, new_xyz = inputs
        # print("idx: ", idx)
        print("QueryBallPtsLayer data : ", grouped_xyz.shape, new_xyz.shape, self.out_dim)
        # TODO: get the idx with self.radius and self.nsample
        rand_data = np.random.randint(1, grouped_xyz.shape[1], self.out_dim).astype(np.float32)
        print("QueryBallPtsLayer res: ", rand_data.shape)

        return [rand_data]


# cv2.dnn_registerLayer('ArgMax', ArgMaxLayer)
cv2.dnn_registerLayer('fps', FPSLayer)
cv2.dnn_registerLayer('idx_pts', IndexPtsLayer)
cv2.dnn_registerLayer('query_ball_pts', QueryBallPtsLayer)
cv2.dnn_registerLayer('sub_center', SubCenterLayer)

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


    def my_fps(g, xyz, npoints):
        return g.op("my_ops::fps", xyz, npoints)

    def my_idx_pts(g, xyz, idx):
        return g.op("my_ops::idx_pts", xyz, idx)

    def my_query_ball_pts(g, radius, nsample, xyz, new_xyz):
        return g.op("my_ops::query_ball_pts", radius, nsample, xyz, new_xyz)

    def my_sub_center(g, grouped_xyz, new_xyz, B, S, C):
        return g.op("my_ops::sub_center", grouped_xyz, new_xyz, B, S, C)


    torch.onnx.register_custom_op_symbolic("my_ops::fps", my_fps, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::idx_pts", my_idx_pts, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::query_ball_pts", my_query_ball_pts, 9)
    torch.onnx.register_custom_op_symbolic("my_ops::sub_center", my_sub_center, 9)

    net = TestNet()
    inputs = torch.randn((1, 3, 1500))
    inputs = inputs.unsqueeze(0)

    out = net(inputs)
    print("**** torch out ******: ", out.shape)
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
                          "res": {0: "b", 1: "n"}
                      }
                      )

    print("onnx model has exported!")

    # inputs = torch.randn((1, 2, 4, 3))
    #
    # center = torch.tensor([[[2.2264, 0.1646, 0.3770], [-0.6141, 0.1071, 2.1032]]])
    # print(center.shape)
    # out = net(inputs, center)
    # print("**** torch out ******: ", out.shape)
    # onnx_path = "./test.onnx"
    # print("start convert model to onnx >>>")
    # torch.onnx.export(net,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
    #                   (inputs, center),
    #                   onnx_path,
    #                   verbose=True,
    #                   input_names=["points", "center"],
    #                   output_names=["res"],
    #                   opset_version=12,
    #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
    #                   # dynamic_axes={
    #                   #     "points": {1: "b", 2: "c", 3: "n"},
    #                   #     "res": {0: "b", 1: "n"}
    #                   # }
    #                   )
    #
    # print("onnx model has exported!")

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
