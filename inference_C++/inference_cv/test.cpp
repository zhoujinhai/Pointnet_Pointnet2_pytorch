#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "fstream"

#include "opencv2/dnn.hpp"
#include <opencv2/dnn/layer.details.hpp>


void printMat(cv::Mat src, int maxC = 15, int maxR = 15) {
    int dim = src.dims;
    std::vector<int> shape;
    for (int i = 0; i < dim; i++) {
        shape.push_back(src.total(i, i + 1));
        std::cout << shape[i] << " ";
    }
    std::cout << " is shape " << std::endl << "data: " << std::endl;

    if (2 == dim) {
        std::cout << src << std::endl;
    }
    else if (3 == dim) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                if (j > maxR) { break; }
                for (int k = 0; k < shape[2]; ++k) {
                    if (k > maxC) { break; }
                    std::cout << src.ptr<float>(i, j)[k] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    else if (4 == dim) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                for (int k = 0; k < shape[2]; ++k) {
                    if (k > maxR) { break; }
                    for (int c = 0; c < shape[3]; ++c) {
                        if (c > maxC) { break; }
                        std::cout << src.ptr<float>(i, j, k)[c] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    else if (5 == dim) {
        std::vector<int> newShape{ shape[1], shape[2], shape[3], shape[4] };
        src = src.reshape(0, newShape);
        for (int i = 0; i < shape[1]; ++i) {
            for (int j = 0; j < shape[2]; ++j) {
                for (int k = 0; k < shape[3]; ++k) {
                    if (k > maxR) { break; }
                    for (int c = 0; c < shape[4]; ++c) {
                        if (c > maxC) { break; }
                        std::cout << src.ptr<float>(i, j, k)[c] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

}


cv::Mat square_distance(cv::Mat src, cv::Mat dst) {
    int B = src.size[0];
    int N = src.size[1];
    int M = dst.size[1];
    std::vector<int> distShape = { B, N, M };
    cv::Mat resDist = cv::Mat::zeros(3, distShape.data(), src.type());
    float* resDistData = (float*)resDist.data;

    // slice to 2D Mat
    cv::Range ranges[3];
    ranges[0] = cv::Range(0, 1);
    ranges[1] = cv::Range::all();
    ranges[2] = cv::Range::all();
    for (int b = 0; b < B; ++b) {
        ranges[0] = cv::Range(b, b + 1);
        cv::Mat slicesrc;
        slicesrc = src(ranges).clone();
        cv::Mat finalSlice(2, &(src.size[1]), src.type());
        slicesrc.copySize(finalSlice);
        //std::cout << slicesrc << std::endl;

        cv::Mat slicedst = dst(ranges).clone();
        cv::Mat finalSlice1(2, &(dst.size[1]), dst.type());
        slicedst.copySize(finalSlice1);
        slicedst = slicedst.t();
        //std::cout << slicedst << std::endl;

        cv::Mat dist = -2.0 * slicesrc * slicedst;
        /* std::cout << "slicexyz: " << slicesrc << std::endl;
         std::cout << "slicepoints: " << slicedst << std::endl;
         std::cout << "dist: " << dist << std::endl;*/

        cv::Mat xyzSum, pointSum;
        cv::reduce(slicesrc.mul(slicesrc), xyzSum, 1, cv::REDUCE_SUM);
        cv::reduce(slicedst.mul(slicedst), pointSum, 0, cv::REDUCE_SUM);

        int dataIdx = 0;

        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < M; ++c) {
                resDistData[dataIdx++] = dist.at<float>(r, c) + xyzSum.at<float>(r) + pointSum.at<float>(c);
            }
        }
    }
    //printMat(resDist);
    return resDist;
}


cv::Mat farthest_point_sampling(cv::Mat xyz, int npoints) {
    int B = xyz.size[0];
    int N = xyz.size[1];
    int C = xyz.size[2];

    cv::Mat centroids = cv::Mat::zeros(B, npoints, CV_32FC1);
    float* centroidsData = (float*)centroids.data;

    for (int b = 0; b < B; ++b) {
        float maxV = xyz.at<float>(b, 0, 0);
        int maxId = 0;
        for (int n = 1; n < N; ++n) {
            if (xyz.at<float>(b, n, 0) > maxV) {
                maxId = n;
                maxV = xyz.at<float>(b, n, 0);
            }
        }
        centroidsData[b * npoints] = maxId;  // first maxId

        cv::Mat distance = cv::Mat::ones(1, N, CV_32FC1) * 1e10;
        for (int i = 1; i < npoints; ++i) {
            // select point  xyz(b, maxId, c) c in 0-C
            cv::Mat tmp = cv::Mat::zeros(1, N, distance.type());
            for (int r = 0; r < N; r++) {
                float sum = 0.0;
                for (int c = 0; c < C; c++) {
                    float sub = xyz.at<float>(b, r, c) - xyz.at<float>(b, maxId, c);
                    sum += sub * sub;
                }
                tmp.at<float>(0, r) = sum;
            }
            
            for (int n = 0; n < N; ++n) {
                distance.at<float>(0, n) = tmp.at<float>(0, n) < distance.at<float>(0, n) ? tmp.at<float>(0, n) : distance.at<float>(0, n);
            }
           
            maxV = distance.at<float>(0, 0);
            maxId = 0;
            for (int n = 1; n < N; ++n) {
                if (distance.at<float>(0, n) > maxV) {
                    maxId = n;
                    maxV = distance.at<float>(0, n);
                }
            }
            centroidsData[b * npoints + i] = maxId;

        }
    }

    return centroids;
}


void farthest_point_sampling(const cv::Mat& xyz, int npoints, cv::Mat& out) {
    int B = xyz.size[0];
    int N = xyz.size[1];
    int C = xyz.size[2];

    float* centroidsData = (float*)out.data;

    for (int b = 0; b < B; ++b) {
        float maxV = xyz.at<float>(b, 0, 0);
        int maxId = 0;
        for (int n = 1; n < N; ++n) {
            if (xyz.ptr<float>(b, n)[0] > maxV) {
                maxId = n;
                maxV = xyz.at<float>(b, n, 0);
            }
        }
        centroidsData[b * npoints] = maxId;  // first maxId

        cv::Mat distance = cv::Mat::ones(1, N, CV_32FC1) * 1e10;
        for (int i = 1; i < npoints; ++i) {
            // select point  xyz(b, maxId, c) c in 0-C
            cv::Mat tmp = cv::Mat::zeros(1, N, distance.type());
            for (int r = 0; r < N; r++) {
                float sum = 0.0;
                for (int c = 0; c < C; c++) {
                    float sub = xyz.ptr<float>(b, r)[c] - xyz.ptr<float>(b, maxId)[c];    // can be optimize by kd_tree
                    sum += sub * sub;
                }
                tmp.at<float>(0, r) = sum;
            }

            for (int n = 0; n < N; ++n) {
                distance.at<float>(0, n) = tmp.at<float>(0, n) < distance.at<float>(0, n) ? tmp.at<float>(0, n) : distance.at<float>(0, n);
            }

            maxV = distance.at<float>(0, 0);
            maxId = 0;
            for (int n = 1; n < N; ++n) {
                if (distance.at<float>(0, n) > maxV) {
                    maxId = n;
                    maxV = distance.at<float>(0, n);
                }
            }
            centroidsData[b * npoints + i] = maxId;

        }
    }
}


cv::Mat query_ball_point(float radius, int nsample, cv::Mat xyz, cv::Mat new_xyz) {
    int B = xyz.size[0];
    int N = xyz.size[1];
    int C = xyz.size[2];
    int S = new_xyz.size[1];
    float refRadius = radius * radius;

    int resShape[] = {B, S, nsample};
    cv::Mat res = cv::Mat::zeros(3, resShape, CV_32FC1);
    float* resData = (float*)res.data;

    cv::Mat dists = square_distance(new_xyz, xyz);
    int groupShape[] = { B, S, N };
    cv::Mat groupIdx = cv::Mat::zeros(3, groupShape, CV_32FC1);  
    float* groupIdxData = (float*)groupIdx.data;

    int groupId = 0;
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < S; ++r) {
            for (int c = 0; c < N; ++c) {
                if (dists.at<float>(b, r, c) > refRadius) {
                    groupIdxData[groupId++] = N;
                }
                else {
                    groupIdxData[groupId++] = c;
                }
            }
        }
    }
    
    //printMat(dists);
    //printMat(groupIdx);

    // slice to 2D Mat
    cv::Range ranges[3];
    ranges[0] = cv::Range(0, 1);
    ranges[1] = cv::Range::all();
    ranges[2] = cv::Range::all();

    int resId = 0;
    for (int b = 0; b < B; ++b) {
        ranges[0] = cv::Range(b, b + 1);

        cv::Mat sliceGroupIdx;
        sliceGroupIdx = groupIdx(ranges).clone();
        cv::Mat newShape(2, &(groupIdx.size[1]), groupIdx.type());
        sliceGroupIdx.copySize(newShape);

        //printMat(sliceGroupIdx);
        cv::sort(sliceGroupIdx, sliceGroupIdx, 0);
        // printMat(sliceGroupIdx);
        
        cv::Mat tempGroup = sliceGroupIdx(cv::Range::all(), cv::Range(0, nsample));
        // printMat(tempGroup);

        for (int r = 0; r < tempGroup.rows; ++r) {
            for (int c = 0; c < tempGroup.cols; ++c) {
                if (tempGroup.at<float>(r, c) == N) {
                    resData[resId++] = tempGroup.at<float>(r, 0);
                }
                else {
                    resData[resId++] = tempGroup.at<float>(r, c);
                }
            }
        }
    }
    printMat(res);

    return res;
}


void query_ball_point(const float radius, const int nsample, const cv::Mat& xyz, const cv::Mat & new_xyz, cv::Mat & out) {
    int B = xyz.size[0];
    int N = xyz.size[1];
    int C = xyz.size[2];
    int S = new_xyz.size[1];
    float refRadius = radius * radius;

    int resShape[] = { B, S, nsample };
    
    float* resData = (float*)out.data;

    cv::Mat dists = square_distance(new_xyz, xyz);
    int groupShape[] = { B, S, N };
    cv::Mat groupIdx = cv::Mat::zeros(3, groupShape, CV_32FC1);
    float* groupIdxData = (float*)groupIdx.data;

    int groupId = 0;
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < S; ++r) {
            for (int c = 0; c < N; ++c) {
                if (dists.at<float>(b, r, c) > refRadius) {
                    groupIdxData[groupId++] = N;
                }
                else {
                    groupIdxData[groupId++] = c;
                }
            }
        }
    }

    // slice to 2D Mat
    cv::Range ranges[3];
    ranges[0] = cv::Range(0, 1);
    ranges[1] = cv::Range::all();
    ranges[2] = cv::Range::all();

    int resId = 0;
    for (int b = 0; b < B; ++b) {
        ranges[0] = cv::Range(b, b + 1);

        cv::Mat sliceGroupIdx;
        sliceGroupIdx = groupIdx(ranges).clone();
        cv::Mat newShape(2, &(groupIdx.size[1]), groupIdx.type());
        sliceGroupIdx.copySize(newShape);

        cv::sort(sliceGroupIdx, sliceGroupIdx, 0);

        cv::Mat tempGroup = sliceGroupIdx(cv::Range::all(), cv::Range(0, nsample));

        for (int r = 0; r < tempGroup.rows; ++r) {
            for (int c = 0; c < tempGroup.cols; ++c) {
                if (tempGroup.at<float>(r, c) == N) {
                    resData[resId++] = tempGroup.at<float>(r, 0);
                }
                else {
                    resData[resId++] = tempGroup.at<float>(r, c);
                }
            }
        }
    }

}


cv::Mat index_points(cv::Mat points, const cv::Mat idx, int channels) {
    int B = points.size[0];
    int C = points.size[2];
    std::vector<int> res_shape;
    res_shape.push_back(B);
    for (int i = 1; i < idx.dims; ++i){
        res_shape.push_back(idx.size[i]);
    }
    res_shape.push_back(C);

    cv::Mat res = cv::Mat::zeros(res_shape.size(), res_shape.data(), points.type());
    float* resData = (float*)res.data;
    float* idxData = (float*)idx.data;
    int idxTotal = idx.total();
    int resIdx = 0;
    //std::cout << "index_points: " << std::endl;
    //printMat(points);
    /*std::cout << "idx: " << std::endl;
    printMat(idx);*/
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < idxTotal; ++i) {
            int selectId = idxData[i] < 0 ? 0 : idxData[i];
            for (int j = 0; j < C; ++j) {
                resData[resIdx++] = points.ptr<float>(b, selectId)[j];
            }
        }
    }
    return res;
}


void index_points(const cv::Mat& points, const cv::Mat& idx, cv::Mat& out, int channels) {
    int B = points.size[0];
    int C = points.size[2];
    float* resData = (float*)out.data;
    float* idxData = (float*)idx.data;
    int idxTotal = idx.total();
    int resIdx = 0;
    //std::cout << "index_points: " << std::endl;
    //printMat(points);
    //std::cout << "idx: " << std::endl;
    //printMat(idx);
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < idxTotal; ++i) {
            int selectId = idxData[i] < 0 ? 0 : idxData[i];
            for (int j = 0; j < C; ++j) {
                resData[resIdx++] = points.ptr<float>(b, selectId)[j];
            }
        }
    }
}


void propagation_data_process(const cv::Mat& xyz1, const cv::Mat& xyz2, const cv::Mat& points1, const cv::Mat& points2, cv::Mat& out) {
    assert(xyz1.size[0] == xyz2.size[0] && xyz2.size[0] == points1.size[0] && points1.size[0] == points2.size[0]);
    cv::Mat dists = square_distance(xyz1, xyz2);  // B * N * M
   
    int B = dists.size[0];
    int N = dists.size[1];
    int M = dists.size[2];
    int shape[] = { B, N, 3 };
    cv::Mat idx = cv::Mat::zeros(3, shape, dists.type());
    float* idxData = (float*)idx.data;
    cv::Mat weights = cv::Mat::zeros(3, shape, dists.type());
    float* weightsData = (float*)weights.data;

    // slice to 2D Mat
    cv::Range ranges[3];
    ranges[0] = cv::Range(0, 1);
    ranges[1] = cv::Range::all();
    ranges[2] = cv::Range::all();

    for (int b = 0; b < B; ++b) {
        ranges[0] = cv::Range(b, b + 1);

        cv::Mat sliceDist;
        sliceDist = dists(ranges).clone();
        cv::Mat newShape(2, &(dists.size[1]), dists.type());
        sliceDist.copySize(newShape);

        cv::Mat sortIdx;
        cv::sortIdx(sliceDist, sortIdx, 2);
        cv::sort(sliceDist, sliceDist, 2);

        sortIdx = sortIdx(cv::Range::all(), cv::Range(0, 3));
        sliceDist = sliceDist(cv::Range::all(), cv::Range(0, 3));
        
        cv::Mat dist_recip = 1.0 / (sliceDist + 1e-8);
        cv::Mat norm;
        cv::reduce(dist_recip, norm, 1, cv::REDUCE_SUM);

        int weightId = 0;
        for (int r = 0; r < dist_recip.rows; ++r) {
            for (int c = 0; c < dist_recip.cols; ++c) {
                weightsData[weightId++] = dist_recip.at<float>(r, c) / norm.at<float>(r);
            }
        }

        int idxId = 0;
        for (int i = 0; i < sortIdx.rows; ++i) {
            for (int j = 0; j < sortIdx.cols; ++j) {
                idxData[idxId++] = sortIdx.at<int>(i, j);
            }
        }
    }

    cv::Mat idxPoints = index_points(points2, idx, points2.size[2]);

    int channels = idxPoints.size[3];
    int weigthTotal = weights.total();
    float* idxPtsData = (float*)idxPoints.data;
    int interPtsId = 0;
    for (int n = 0; n < weigthTotal; ++n) {
        for (int c = 0; c < channels; ++c) {
            idxPtsData[interPtsId] = idxPtsData[interPtsId] * weightsData[n];
            interPtsId++;
        }
    }

    // torch::sum 2
    int delDim = idxPoints.size[2];
    std::vector<int>idxPointsSize = { idxPoints.size[0], idxPoints.size[1], idxPoints.size[3] };
    cv::Mat interpolatedPts = cv::Mat::zeros(idxPointsSize.size(), idxPointsSize.data(), idxPoints.type());
    float* interpolatedPtsData = (float*)interpolatedPts.data;
    interPtsId = 0;
    for (int i = 0; i < idxPointsSize[0]; ++i) {
        for (int j = 0; j < idxPointsSize[1]; ++j) {
            for (int k = 0; k < idxPointsSize[2]; ++k) {
                for (int n = 0; n < delDim; ++n) {
                    interpolatedPtsData[interPtsId] += idxPoints.ptr<float>(i, j, n)[k];
                }
                interPtsId++;
            }
        }
    }

    // cat points1 interpolated_points
    assert(points1.size[0] == interpolatedPts.size[0] && points1.size[2] == interpolatedPts.size[1]);

    std::vector<int> catShape;
    catShape.push_back(points1.size[0]);
    catShape.push_back(points1.size[2]);
    catShape.push_back(points1.size[1] + interpolatedPts.size[2]);

    float* catData = (float*)out.data;
    int catId = 0;
    for (int b = 0; b < catShape[0]; ++b) {
        for (int n = 0; n < catShape[1]; ++n) {
            for (int r = 0; r < points1.size[1]; ++r) {
                catData[catId++] = points1.at<float>(b, r, n);
            }
            for (int c = 0; c < interpolatedPts.size[2]; ++c) {
                catData[catId++] = interpolatedPts.ptr<float>(b, n)[c];
            }
        }
    }
}


bool readFile(std::string dataPath, std::vector<std::vector<float> >& points, bool useNormal = false) {

    int cols = 3;
    if (useNormal) {
        cols = 6;
    }
    std::fstream infile;
    infile.open(dataPath, std::ios::in);
    if (!infile.is_open()) {
        std::cout << "open file " << dataPath << " failed! " << std::endl;
        return false;
    }
    std::vector<std::string> data;
    std::string s;
    while (std::getline(infile, s)) {
        data.push_back(s);
    }

    std::string suffix = dataPath.substr(dataPath.find_last_of('.') + 1);
    if (suffix == "txt" || suffix == "pts") {
        for (int i = 0; i < data.size(); ++i) {
            std::vector<float> line_data;
            std::istringstream ss(data[i]);
            std::string item;
            int n = 0;
            while (n < cols && ss >> item) {
                line_data.push_back(std::stof(item));
                n += 1;
            }
            points.push_back(line_data);
        }
    }
    else if (suffix == "pcd") {
        int skipRows = 10;
        for (int i = skipRows; i < data.size(); ++i) {
            std::vector<float> line_data;
            std::istringstream ss(data[i]);
            std::string item;
            int n = 0;
            while (n < cols && ss >> item) {
                line_data.push_back(std::stof(item));
                n += 1;
            }
            points.push_back(line_data);
        }
    }
    else {
        std::cout << "file type don't supported, just txt or pcd!";
        return false;
    }

    return true;
}


bool PreProcess(std::vector<std::vector<float> >& points, std::vector<int>& ids)
{
    // 归一化
    int n = points.size();
    if (n < 1) {
        return false;
    }

    float x_mean = 0.0, y_mean = 0.0, z_mean = 0.0;
    for (int i = 0; i < n; ++i) {
        x_mean += points[i][0];
        y_mean += points[i][1];
        z_mean += points[i][2];
    }
    x_mean /= n;
    y_mean /= n;
    z_mean /= n;

    for (int i = 0; i < n; ++i) {
        points[i][0] -= x_mean;
        points[i][1] -= y_mean;
        points[i][2] -= z_mean;
    }

    float maxSqrtSquare = 0.0;
    for (int i = 0; i < n; ++i) {
        float sqrtSquare = std::sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1] + points[i][2] * points[i][2]);
        if (sqrtSquare > maxSqrtSquare) {
            maxSqrtSquare = sqrtSquare;
        }
    }
    
    for (int i = 0; i < n; ++i) {
        points[i][0] /= maxSqrtSquare;
        points[i][1] /= maxSqrtSquare;
        points[i][2] /= maxSqrtSquare;
        ids.push_back(i);
    }

    // 随机打乱
    std::random_shuffle(ids.begin(), ids.end());
    std::vector<std::vector<float> > newPoints;
    for (int i = 0; i < n; ++i) {
        newPoints.push_back(points[ids[i]]);
    }
    points.swap(newPoints);
}


class ArgMaxLayer CV_FINAL : public cv::dnn::Layer
{
public:
    ArgMaxLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        axis = params.get<int>("axis", 0);   // axis may be negative in python, there is not consider it.
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new ArgMaxLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> input = inputs[0];
        int inputShape = input.size();
        std::vector<int> outShape;
        for (int i = 0; i < inputShape; ++i) {
            if (i == axis) { continue; }
            outShape.push_back(input[i]);
        }
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            // In case of DNN_TARGET_OPENCL_FP16 target the following method
            // converts data from FP16 to FP32 and calls this forward again.
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];   // B * N * 3
        cv::Mat& out = outputs[0];
        float* outData = (float*)out.data;
         
        int dim = inp.dims;
        std::vector<int> inSize;
        for (int i = 0; i < dim; i++) {
            inSize.push_back(inp.total(i, i + 1));
        }
        
        // Just implement axis = 1 or 2 for 3D Mat and axis = 1 for 2D Mat
        int batch = inSize[0];
        for (int b = 0; b < batch; ++b) {
            if (dim == 2) {
                int maxId = 0;
                float maxV = inp.ptr<float>(b)[0];
                for (int j = 1; j < inSize[axis]; ++j) {
                    if (inp.ptr<float>(b)[j] > maxV) {
                        maxV = inp.ptr<float>(b)[j];
                        maxId = j;
                    }
                }
                outData[b] = float(maxId);
            }
            else if (dim == 3) {
                int n = inSize[3 - axis];
                for (int i = 0; i < n; ++i) {
                    int maxId = 0;
                    if (axis == 1) {
                        float maxV = inp.ptr<float>(b, 0)[i];  // T.ptr<float>(b, :)[i]
                        for (int j = 1; j < inSize[axis]; ++j) {
                            if (inp.ptr<float>(b, j)[i] > maxV) {
                                maxV = inp.ptr<float>(b, j)[i];
                                maxId = j;
                            }
                        }
                        outData[b * n + i] = float(maxId);
                    }
                    else if (axis == 2) {
                        float maxV = inp.ptr<float>(b, i)[0];  // T.ptr<float>(b, i)[:]
                        for (int j = 1; j < inSize[axis]; ++j) {
                            if (inp.ptr<float>(b, i)[j] > maxV) {
                                maxV = inp.ptr<float>(b, i)[j];
                                maxId = j;
                            }
                        }
                        outData[b * n + i] = float(maxId);
                    }
                }
            }
        }
        
    }
private:
    int axis;
};


class FPSLayer CV_FINAL : public cv::dnn::Layer
{
public:
    FPSLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        cv::Mat blob = params.blobs[0];
        nPoint = blob.at<int>(0);
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new FPSLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> input = inputs[0];
        std::vector<int> outShape;
        const int batch = input[0];
        outShape.push_back(batch);
        outShape.push_back(nPoint);
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            // In case of DNN_TARGET_OPENCL_FP16 target the following method
            // converts data from FP16 to FP32 and calls this forward again.
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];   // B * N * 3
        cv::Mat& out = outputs[0];

        // implement fps by C++
        farthest_point_sampling(inp, nPoint, out);  // need to optimize
        //printMat(out);
    }
private:
    int nPoint;
};


class IndexPtsLayer CV_FINAL : public cv::dnn::Layer
{
public:
    IndexPtsLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        channel = blobs[0].at<int>(0);
    }
    
    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new IndexPtsLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> input = inputs[0];
        std::vector<int> idx = inputs[1];

        std::vector<int> outShape;
        outShape.push_back(1);
        const int batch = input[0];
        outShape.push_back(batch);
        for (int i = 1; i < idx.size(); ++i) {
            outShape.push_back(idx[i]);
        }
        outShape.push_back(channel);  // input[2]
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];   // B * N * 3
        cv::Mat& idx = inputs[1];   // B * S
        cv::Mat& out = outputs[0];
        index_points(inp, idx, out, inp.size[2]);
        //printMat(out);
    }
private:
    int channel;
};


class QueryBallPtsLayer CV_FINAL : public cv::dnn::Layer
{
public:
    QueryBallPtsLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        std::vector<cv::Mat> blobs = params.blobs;
        radius = blobs[0].at<float>(0);  // blobs[0]  nPoint = blob.at<int>(0);
        nsample = blobs[1].at<int>(0);   // blobs[1]
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new QueryBallPtsLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> xyz = inputs[0];
        std::vector<int> new_xyz = inputs[1];

        std::vector<int> outShape;
        outShape.push_back(1);
        const int batch = xyz[0];
        outShape.push_back(batch);
        outShape.push_back(new_xyz[1]);
        outShape.push_back(nsample);
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& xyz = inputs[0];   // B * N * 3
        cv::Mat& new_xyz = inputs[1];   // B * S * 3
        cv::Mat& out = outputs[0];

        //Implement the index points
        query_ball_point(radius, nsample, xyz, new_xyz, out);
        //printMat(out);
    }
private:
    float radius;
    int nsample;
};


class SubCenterLayer CV_FINAL : public cv::dnn::Layer
{
public:
    SubCenterLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new SubCenterLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> grouped_xyz = inputs[0];
        std::vector<int> new_xyz = inputs[1];

        outputs.assign(1, grouped_xyz);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& grouped_xyz = inputs[0];   // B * S * N * 3
        cv::Mat& new_xyz = inputs[1];   // B * S * 3
        cv::Mat& out = outputs[0];
        float* outData = (float*)out.data;

        int dim = grouped_xyz.dims;
        std::vector<int> groupedShape;
        for (int i = 0; i < dim; i++) {
            groupedShape.push_back(grouped_xyz.total(i, i + 1));
        }

        int new_dim = new_xyz.dims;
        std::vector<int> newShape;
        for (int i = 0; i < new_dim; i++) {
            newShape.push_back(new_xyz.total(i, i + 1));
        }

        // assert dim and shape
        assert(dim == 4 && new_dim == 3);
        assert(groupedShape[0] == newShape[0] && groupedShape[1] == newShape[1] && groupedShape[3] == newShape[2]);

        // get shape, TODO: optimize like numpy broadcast, may be can use eigen3
        int B = groupedShape[0];
        int N = groupedShape[1];
        int S = groupedShape[2];
        int C = groupedShape[3];

        // deal data
        int i = 0;
        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < S; ++s) {
                    for (int c = 0; c < C; ++c) {
                        outData[i++] = grouped_xyz.ptr<float>(b, n, s)[c] - new_xyz.ptr<float>(b, n)[c];
                    }
                }
            }
        }

        //printMat(out);
    }
};


class TileLayer CV_FINAL : public cv::dnn::Layer
{
public:
    TileLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        cv::Mat blob = blobs[0];
        int n = blob.rows;
        for (int i = 0; i < n; ++i) {
            if (blob.at<int>(i, 0) != 1) {
                idx = i;
                nRepeat = blob.at<int>(i, 0);
                break;
            }
        }
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new TileLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> input_dim = inputs[0];
        input_dim[idx] = nRepeat;

        outputs.assign(1, input_dim);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& input = inputs[0];   
        
        cv::Mat& out = outputs[0];
        float* outData = (float*)out.data;

        int dim = input.dims;
        // std::cout << input.size[0] << input.size[1] << input.size[2] << std::endl;
        std::vector<int> inputShape;
        for (int i = 0; i < dim; i++) {
            inputShape.push_back(input.size[i]);  //input.total(i, i + 1);
        }

        int B = inputShape[0];
        int C = inputShape[1];
        int N = inputShape[2];
        int i = 0;
        for (int n = 0; n < nRepeat; n++) {
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < C; c++) {
                    for (int j = 0; j < N; j++) {
                        outData[i++] = input.ptr<float>(b, c)[j];   // just implement (1, 1, 1024) ==> (1, 256, 1024)
                    }
                }
            }
        }
        /*std::cout << "input: " << std::endl;
        printMat(input);
        std::cout << "out: " << std::endl;*/
        // printMat(out);
    }
private:
    int idx;
    int nRepeat;
};


class PropagateDataLayer CV_FINAL : public cv::dnn::Layer
{
public:
    PropagateDataLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
       
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new PropagateDataLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> xyz1 = inputs[0];
        std::vector<int> xyz2 = inputs[1];
        std::vector<int> points1 = inputs[2];
        std::vector<int> points2 = inputs[3];
        int out_channel = int(points1[1] + points2[2]);

        std::vector<int> out_dim;
        out_dim.push_back(1);
        out_dim.push_back(xyz1[0]);
        out_dim.push_back(xyz1[1]);
        out_dim.push_back(out_channel);

        outputs.assign(1, out_dim);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& xyz1 = inputs[0];  
        cv::Mat& xyz2 = inputs[1];
        cv::Mat& points1 = inputs[2];
        cv::Mat& points2 = inputs[3];

        cv::Mat& out = outputs[0];
       
        propagation_data_process(xyz1, xyz2, points1, points2, out);
        // printMat(out);
    }
};


class GetCateLayer CV_FINAL : public cv::dnn::Layer
{
public:
    GetCateLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        cv::Mat blob1 = blobs[0];
        cv::Mat blob2 = blobs[1];
        n_cls = blob1.at<int>(0);
        int C = blob2.at<int>(0);
        channel = n_cls + 2 * C;
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new GetCateLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        std::vector<int> src = inputs[0];
        std::vector<int> dst = inputs[1];

        std::vector<int> out_dim;
        out_dim.push_back(1);
        out_dim.push_back(src[0]);
        out_dim.push_back(channel);
        out_dim.push_back(src[2]);

        outputs.assign(1, out_dim);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& xyz = inputs[0];
        cv::Mat& points = inputs[1];

        cv::Mat& out = outputs[0];
        float* outData = (float*)out.data;

        int dim = xyz.dims;
        std::vector<int> xyzShape;
        for (int i = 0; i < dim; i++) {
            xyzShape.push_back(xyz.total(i, i + 1));
        }

        int dim1 = points.dims;
        std::vector<int> pointsShape;
        for (int i = 0; i < dim1; i++) {
            pointsShape.push_back(points.total(i, i + 1));
        }

        assert(dim == dim1 && xyzShape[0] == pointsShape[0] && xyzShape[2] == pointsShape[2]) ;

        int B = xyzShape[0];
        int nXYZ = xyzShape[1];
        int N = xyzShape[2];
        int nPts = pointsShape[1];

        int i = 0;
        for (int c = 0; c < n_cls * N; ++c) {
            outData[i++] = 1.0;
        }
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < nXYZ; ++j) {
                for (int k = 0; k < N; ++k) {
                    outData[i++] = xyz.ptr<float>(b, j)[k];
                }
            }
            for (int j = 0; j < nPts; ++j) {
                for (int k = 0; k < N; ++k) {
                    outData[i++] = points.ptr<float>(b, j)[k];
                }
            }
        }
        /*printMat(xyz);
        printMat(points);*/

        // printMat(out);

    }
private:
    int channel;
    int n_cls;
};


int main(int argc, char* argv[])
{

    ////std::vector<int> xyzShape{ 1, 10, 3 };

    ////// cv::Mat a = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);

    /////*std::vector<float> testData;
    ////for (int i = 1; i < 31; ++i) {
    ////    testData.push_back(i);
    ////}
    ////cv::Mat xyz(testData);
    ////xyz = xyz.reshape(0, xyzShape);*/

    //cv::Mat xyz1 = (cv::Mat_<float>(8, 3) << 0.2901, 1.2105, -1.2701, 1.6728, 0.1126, 0.2237, -0.2826, -0.4668, -0.7778, 0.3933, -1.8726, -0.2206,
    //    0.8394, 0.6340, -0.3097, 0.2373, -1.2495, 0.7897, 0.1440, -0.8319, -2.0404, -1.2246, 0.3089, -0.0196);
    //xyz1 = xyz1.reshape(0, { 1, 8, 3 });

    //cv::Mat xyz2 = (cv::Mat_<float>(5, 3) << -0.0043, 1.3982, -1.2204, -0.6885, 0.4035, -2.2926,
    //    -0.9572, 0.5397, -1.3338, 1.5315, -1.9127, -0.2904, 1.4844, -0.5510, -2.4500);
    //xyz2 = xyz2.reshape(0, { 1, 3, 5 });

    //printMat(xyz1);
    //printMat(xyz2);

    //std::vector<int> catShape;
    //catShape.push_back(xyz1.size[0]);
    //catShape.push_back(xyz1.size[2]);
    //catShape.push_back(xyz1.size[1] + xyz2.size[2]);

    //std::vector<int> testShape = { 1, 1, 3, 13 };
    //cv::Mat testRes(4, testShape.data(), xyz1.type());
    //float* catData = (float*)testRes.data;
    //int catId = 0;
    //for (int b = 0; b < catShape[0]; ++b) {
    //    for (int n = 0; n < catShape[1]; ++n) {
    //        for (int r = 0; r < xyz1.size[1]; ++r) {
    //            catData[catId++] = xyz1.at<float>(b, r, n);
    //        }
    //        for (int c = 0; c < xyz2.size[2]; ++c) {
    //            catData[catId++] = xyz2.ptr<float>(b, n)[c];
    //        }
    //    }
    //}
    //printMat(testRes);
    //
    //std::cout << "$$$$$$$$" << std::endl;
    
    
    // load data
    std::string dataPath = "D:/Debug_dir/inputs.pts";  // "D:/Debug_dir/news_data/pcd_label_normal/bankou (1)_minCruv.pcd";  // argv[1]

    std::vector<std::vector<float> > points;                // N * 3
    if (!readFile(dataPath, points)) {
        return -1;
    }
    
    // preprocess data
    std::vector<int> ids;
    /*if (!PreProcess(points, ids)) {
        return -1;
    }*/
    for (int i = 0; i < points.size(); ++i) {
        ids.push_back(i);
    }

    cv::Mat datas(points.at(0).size(), points.size(), CV_32FC1);  // 3 * N
    for (int i = 0; i < datas.rows; ++i) {
        for (int j = 0; j < datas.cols; ++j) {
            datas.at<float>(i, j) = points.at(j).at(i);
        }
    }       
   
    std::vector<int> newShape{1, 1, datas.rows, datas.cols};
    datas = datas.reshape(0, newShape);    // 1 * 1 * 3 * N
    printMat(datas);
    //// test
    //int size[3] = { 1, 2, 3 };
    //cv::Mat mat3D(3, size, CV_8UC1, cv::Scalar::all(0));
    
    /*float a = datas.ptr<float>(0, 0, 0)[0];
    float b = datas.ptr<float>(0, 0, 1)[0];
    float c = datas.ptr<float>(0, 0, 2)[0];*/

    // load model
    
    const std::string modelPath = "E:/code/Server223/pointNet/inference_C++/test.onnx";
    //CV_DNN_REGISTER_LAYER_CLASS(ArgMax, ArgMaxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(fps, FPSLayer);
    CV_DNN_REGISTER_LAYER_CLASS(idx_pts, IndexPtsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(query_ball_pts, QueryBallPtsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(sub_center, SubCenterLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Tile, TileLayer);
    CV_DNN_REGISTER_LAYER_CLASS(propagatedata, PropagateDataLayer);
    CV_DNN_REGISTER_LAYER_CLASS(get_cate, GetCateLayer);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    net.setInput(datas, "points");

    cv::Mat res = net.forward();
    
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = endTime - startTime;
    std::cout << "all code took " << time_span.count() << " seconds." << std::endl;

    int nRes = res.total();
    printMat(res);
    std::cout << " total item: " << nRes << std::endl;

    int B = res.size[0];  // batch number: Here is 1.
    int N = res.size[1];  // points number
    int C = res.size[2];  // class number: Here seg to two class, the value is 2.

    // parse result
    std::vector<int> resIds;
    std::vector<std::vector<float> > resPts;
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            if (res.at<float>(b, n, 0) < res.at<float>(b, n, 1)) {
                resPts.push_back(points[n]);
                resIds.push_back(ids[n]);
            }
        }
    }

    std::fstream outfile;
    outfile.open("D:/Debug_dir/res.pts", std::ios::out);
    if (!outfile.is_open()) {
        std::cout << "write res to file is failed! " << std::endl;
        return false;
    }
    for (auto pt : resPts) {
        outfile << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    outfile.close();

    std::fstream outfile1;
    outfile1.open("D:/Debug_dir/test.pts", std::ios::out);
    if (!outfile1.is_open()) {
        std::cout << "write test to file is failed! " << std::endl;
        return false;
    }
    for (auto pt : points) {
        outfile1 << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    outfile1.close();
    
    return 0;
}