#include "torch/script.h"
#include <iostream>


/* python code
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
        mask = dist < distance
        distance[mask] = dist[mask]  # torch.where(mask, dist, distance)
        farthest = torch.max(distance, -1)[1]

    return centroids

*/
torch::Tensor farthest_point_sampling(torch::Tensor xyz, long long npoint) {
    /*
     Input:
        xyz: pointcloud data, [B, N, 3]
        nSamples : number of samples
     Return :
        centroids: sampled pointcloud index, [B, npoint]
    */
    torch::Device device = xyz.device();
    torch::Tensor output = torch::ones({ xyz.size(0), npoint }, at::device(device).dtype(at::ScalarType::Long));
    torch::Tensor distance = torch::full({ xyz.size(0), xyz.size(1) }, 1e10, at::device(device).dtype(at::ScalarType::Float));

    torch::Tensor farthest = std::get<1>(torch::max(xyz.index({ "...", 0 }), 1));

    torch::Tensor batchIndices = torch::arange(0, xyz.size(0)).to(device);
    for (int i = 0; i < npoint; ++i) {
        output.slice(1, i, i+1) = farthest.unsqueeze(1);
        torch::Tensor centroid = xyz.index({ batchIndices, farthest, "..." }).view({ xyz.size(0), 1, 3});
        torch::Tensor dist =  torch::sum((xyz - centroid) * (xyz - centroid), -1);
        distance = torch::_s_where(dist < distance, dist, distance);
        farthest = std::get<1>(torch::max(distance, -1));
    }

    return output.clone();
}


//def index_points(points, idx) :
//    """
//
//    Input :
//    points : input points data, [B, N, C]
//    idx : sample index data, [B, S]
//    Return :
//    new_points : , indexed points data, [B, S, C]
//    """
//    device = points.device
//    B = points.shape[0]
//    # B = int(B)
//    view_shape = list(idx.shape)
//    # view_shape[1:] = [1] * (len(view_shape) - 1)
//    for i in range(1, len(view_shape)) :
//        view_shape[i] = 1
//    repeat_shape = list(idx.shape)
//    repeat_shape[0] = 1
//    print("view_shape: ", view_shape, " repeat_shape: ", repeat_shape)
//    batch_indices = torch.arange(B, dtype = torch.long, device = device).view(view_shape).repeat(repeat_shape)
//    new_points = points[batch_indices, idx, :]
//    return new_points
torch::Tensor index_points(torch::Tensor points, torch::Tensor idx, long long channels) {
    /*
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] / [B, S1, S2]
    Return:
        new_points:, indexed points data, [B, S, C] / [B, S1, S2, C]
    */
    torch::Device device = points.device();
    int B = points.size(0);
    std::vector<int64_t> view_shape = idx.sizes().vec();
    for (int i = 1; i < view_shape.size(); ++i) {
        view_shape[i] = 1;
    }

    std::vector<int64_t> repeat_shape = idx.sizes().vec();
    repeat_shape[0] = 1;
    // std::cout << " repeat_shape: " << repeat_shape << std::endl;
    torch::Tensor batch_indices = torch::arange(B, at::device(device).dtype(at::ScalarType::Long)).view(view_shape).repeat(repeat_shape);
    // std::cout << batch_indices << std::endl;
    torch::Tensor new_points = points.index({ batch_indices, idx, "..." });
    // std::cout << new_points << std::endl;
    new_points = new_points.unsqueeze(0);  // for opencv, it not support 3D output
    return new_points.clone();
}


//def square_distance(src, dst) :
//    """
//    Calculate Euclid distance between each two points.
//
//    src ^ T * dst = xn * xm + yn * ym + zn * zm；
//    sum(src ^ 2, dim = -1) = xn * xn + yn * yn + zn * zn;
//sum(dst ^ 2, dim = -1) = xm * xm + ym * ym + zm * zm;
//dist = (xn - xm) ^ 2 + (yn - ym) ^ 2 + (zn - zm) ^ 2
//= sum(src * *2, dim = -1) + sum(dst * *2, dim = -1) - 2 * src ^ T * dst
//
//Input :
//src: source points, [B, N, C]
//dst : target points, [B, M, C]
//Output :
//    dist : per - point square distance, [B, N, M]
//    """
//    B, N, _ = src.shape
//    _, M, _ = dst.shape
//    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
//    dist += torch.sum(src * *2, -1).view(B, N, 1)
//    dist += torch.sum(dst * *2, -1).view(B, 1, M)
//    return dist
//
//
//    def query_ball_point(radius: float, nsample : int, xyz, new_xyz) :
//    """
//    Input :
//    radius : local region radius
//    nsample : max sample number in local region
//    xyz : all points, [B, N, 3]
//    new_xyz : query points, [B, S, 3]
//    Return :
//    group_idx : grouped points index, [B, S, nsample]
//    """
//    device = xyz.device
//    B, N, C = xyz.shape
//    _, S, _ = new_xyz.shape
//    group_idx = torch.arange(N, dtype = torch.long, device = device).view(1, 1, N).repeat([B, S, 1])
//    sqrdists = square_distance(new_xyz, xyz)
//    # mask = sqrdists > radius * *2
//    # group_idx[mask] = N
//    temp = torch.full((B, S, N), N, dtype = torch.long)  # torch.ones(group_idx.shape, dtype = torch.long) * N
//    group_idx = torch.where(sqrdists > radius * *2, temp, group_idx)
//
//    group_idx = group_idx.sort()[0][:, : , : nsample]
//    group_first = group_idx[:, : , 0].view(B, S, 1).repeat([1, 1, nsample])
//    # mask = group_idx == N
//    # group_idx[mask] = group_first[mask]
//    # print("N: ", N)
//    group_idx = torch.where(group_idx == N, group_first, group_idx)
//    return group_idx
torch::Tensor square_distance(torch::Tensor src, torch::Tensor dst) {
    int B = src.size(0);
    int N = src.size(1);
    int M = dst.size(1);
    torch::Tensor dist = -2.0 * torch::matmul(src, dst.permute({ 0, 2, 1 }));
    dist += torch::sum(src * src, -1).view({ B, N, 1 });
    dist += torch::sum(dst * dst, -1).view({ B, 1, M });
    dist = dist.unsqueeze(0);
    return dist.clone();
}


//def query_ball_point(radius: float, nsample : int, xyz, new_xyz) :
//    """
//    Input :
//    radius : local region radius
//    nsample : max sample number in local region
//    xyz : all points, [B, N, 3]
//    new_xyz : query points, [B, S, 3]
//    Return :
//    group_idx : grouped points index, [B, S, nsample]
//    """
//    device = xyz.device
//    B, N, C = xyz.shape
//    _, S, _ = new_xyz.shape
//    B, N, C, S = int(B), int(N), int(C), int(S)
//    group_idx = torch.arange(N, dtype = torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
//    sqrdists = square_distance(new_xyz, xyz)
//    # mask = sqrdists > radius * *2
//    # group_idx[mask] = N
//    temp = torch.ones(group_idx.shape, dtype = torch.long) * N    # (B, S, N)
//    group_idx = torch.where(sqrdists > radius * *2, temp, group_idx)
//
//    group_idx = group_idx.sort(dim = -1)[0][:, : , : nsample]
//    group_first = group_idx[:, : , 0].view(B, S, 1).repeat([1, 1, nsample])
//    # mask = group_idx == N
//    # group_idx[mask] = group_first[mask]
//    group_idx = torch.where(group_idx == N, group_first, group_idx)
//    return group_idx
torch::Tensor query_ball_point(torch::Tensor radius, long long nsample, torch::Tensor xyz, torch::Tensor new_xyz) {
    at::Device device = xyz.device();
    int B = xyz.size(0);
    int N = xyz.size(1);
    int C = xyz.size(2);
    int S = new_xyz.size(1);
    float r = radius.item().toFloat();
    float refRadius = r * r;

    torch::Tensor group_idx = torch::arange(N, at::device(device).dtype(at::ScalarType::Long)).view({ 1, 1, N }).repeat({ B, S, 1 });
    torch::Tensor sqrdists = square_distance(new_xyz, xyz).squeeze(0);
    //std::cout << "sqrdists: " << sqrdists << std::endl;
    
    torch::Tensor temp = torch::full({ B, S, N }, N, dtype(at::ScalarType::Long));
    group_idx = torch::_s_where(sqrdists > refRadius, temp, group_idx);

    torch::Tensor temp_sort = std::get<0>(group_idx.sort());
    //std::cout << "temp_sort: " << temp_sort << std::endl;
    group_idx = temp_sort.index({ "...", torch::indexing::Slice(0, nsample) }); 

    /* deal group_idx may be N due to group_first is N
    torch::Tensor dist_idx = std::get<1>(torch::sort(sqrdists, 2));
    torch::Tensor temp_first = group_idx.index({ "...", 0 });
    temp_first = torch::_s_where(temp_first == N, dist_idx.index({ "...", 0 }), temp_first);

    torch::Tensor group_first = temp_first.view({ B, S, 1 }).repeat({ 1, 1, nsample });*/
    
    //std::cout << "group_idx: " << group_idx << std::endl;
    torch::Tensor group_first = group_idx.index({ "...", 0 }).view({ B, S, 1 }).repeat({ 1, 1, nsample });

    group_idx = torch::_s_where(group_idx == N, group_first, group_idx);
    group_idx = group_idx.unsqueeze(0);  // for opencv, it not support 3D output
    return group_idx.clone();
}


torch::Tensor sub_center(torch::Tensor grouped_xyz, torch::Tensor new_xyz) {
    grouped_xyz -= new_xyz.view({ new_xyz.size(0), new_xyz.size(1), 1, new_xyz.size(2) });
    return grouped_xyz;
}


torch::Tensor get_cate(torch::Tensor xyz, torch::Tensor points, long long nCate, long long C) {
    torch::Tensor cate = torch::ones({ xyz.size(0), nCate, xyz.size(2) });
    torch::Tensor res = torch::cat({ cate, xyz, points }, 1);
    res = res.unsqueeze(0);
    return res.clone();
}


torch::Tensor propagation_data_process(torch::Tensor xyz1, torch::Tensor xyz2, torch::Tensor points1, torch::Tensor points2) {
    torch::Tensor dists = square_distance(xyz1, xyz2).squeeze(0);
    std::tuple<torch::Tensor, torch::Tensor> sorted = dists.sort(2);
    dists = std::get<0>(sorted);
    torch::Tensor idx = std::get<1>(sorted);
    
    dists = dists.index({ "...", torch::indexing::Slice(0, 3) });
    idx = idx.index({ "...", torch::indexing::Slice(0, 3) });

    torch::Tensor dist_recip = 1.0 / (dists + 1e-8);
    torch::Tensor norm = torch::sum(dist_recip, 2, true);
    torch::Tensor weight = dist_recip / norm;
    
    torch::Tensor interpolated_points = torch::sum(index_points(points2, idx, points2.size(2)).squeeze(0) * weight.view({ xyz1.size(0), xyz1.size(1), 3, 1 }), 2);

    points1 = points1.permute({ 0, 2, 1 });
    torch::Tensor new_points = torch::cat({ points1, interpolated_points }, -1);

    new_points = new_points.unsqueeze(0);
    return new_points.clone();
}


at::TensorList sample_and_group_all(torch::Tensor xyz, torch::Tensor points) {
    at::Device device = xyz.device();
    int B = xyz.size(0);
    int N = xyz.size(1);
    int C = xyz.size(2);

    torch::Tensor new_xyz = torch::zeros({ B, 1, C }, at::device(device));
    torch::Tensor grouped_xyz = xyz.view({ B, 1, N, C });
    std::vector<torch::Tensor> tempVec;
    tempVec.push_back(grouped_xyz);
    tempVec.push_back(points.view({ B, 1, N, -1 }));
    at::TensorList tempList(tempVec);
    torch::Tensor new_points = torch::cat(tempList, -1);   // 可提前一步返回,后续再拼接

    tempVec.clear();
    tempVec.push_back(new_xyz);
    tempVec.push_back(new_points);
    return tempVec;
}


// torch.__version__: 1.5.0
 //static auto registry = torch::RegisterOperators("my_ops::fps", &farthest_point_sampling);  
 //static auto registry = torch::RegisterOperators("my_ops::idx_pts", &index_points);
 static auto registry = torch::RegisterOperators("my_ops::query_ball_pts", &query_ball_point);
 //static auto registry = torch::RegisterOperators("my_ops::sub_center", &sub_center);
// static auto registry = torch::RegisterOperators("my_ops::sample_and_group_all", &sample_and_group_all);
 //static auto registry = torch::RegisterOperators("my_ops::get_cate", &get_cate); 
 //static auto registry = torch::RegisterOperators("my_ops::dist", &square_distance);
//static auto registry = torch::RegisterOperators("my_ops::propagatedata", &propagation_data_process);

//// torch.__version__ >= 1.6.0  torch/include/torch/library.h
//TORCH_LIBRARY(my_ops, m) {
//   m.def("fps", farthest_point_sampling);
//}

// int main(int argc, char* argv[]) {
//     torch::Tensor xyz = torch::rand({ 1, 50, 3 });
//     std::cout << "xyz: " << xyz << std::endl;
//
//     long long n = 3;
//     torch::Tensor idx = farthest_point_sampling(xyz, n);
//     std::cout << idx << std::endl;
//
//     torch::Tensor new_xyz = index_points(xyz, idx).squeeze(0);
//     std::cout << new_xyz << std::endl;
//
//     torch::Tensor group_idx = query_ball_point(torch.tensor(0.1), 32, xyz, new_xyz);
//     std::cout << group_idx << std::endl;
//     return 0;
//}
