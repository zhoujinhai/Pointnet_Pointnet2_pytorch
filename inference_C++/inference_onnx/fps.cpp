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


 static auto registry = torch::RegisterOperators("my_ops::fps", &farthest_point_sampling);  // torch.__version__: 1.5.0

//// torch.__version__ >= 1.6.0  torch/include/torch/library.h
//TORCH_LIBRARY(my_ops, m) {
//   m.def("fps", farthest_point_sampling);
//}

// int main(int argc, char* argv[]) {
//     torch::Tensor xyz = torch::rand({ 2, 10, 3 });
//     std::cout << "xyz: " << xyz << std::endl;
//
//     long long n = 2;
//     torch::Tensor out = farthest_point_sampling(xyz, n);
//     std::cout << out << std::endl;
//    
//     return 0;
//}