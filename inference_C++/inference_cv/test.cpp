#include <iostream>
#include <vector>
#include <chrono>
#include "fstream"

#include "opencv2/dnn.hpp"
#include <opencv2/dnn/layer.details.hpp>


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
    if (suffix == "txt") {
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
        float* outData = (float*)out.data;

        int dim = inp.dims;
        std::vector<int> inSize;
        for (int i = 0; i < dim; i++) {
            inSize.push_back(inp.total(i, i + 1));
        }

        // TODO implement fps by C++

    }
private:
    int nPoint;
};


class IndexPtsLayer CV_FINAL : public cv::dnn::Layer
{
public:
    IndexPtsLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {

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
        outShape.push_back(3);  // input[2]
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
        float* outData = (float*)out.data;

        int dim = inp.dims;
        std::vector<int> inSize;
        for (int i = 0; i < dim; i++) {
            inSize.push_back(inp.total(i, i + 1));
        }

        // TODO Implement the index points
    }
};


class QueryBallPtsLayer CV_FINAL : public cv::dnn::Layer
{
public:
    QueryBallPtsLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        std::cout << params << std::endl;
        std::vector<cv::Mat> blobs = params.blobs;
        radius = blobs[0].at<float>(0);  // blobs[0]  nPoint = blob.at<int>(0);
        nsample = blobs[1].at<int>(0);   // blobs[1]
        std::cout << "radius: " << radius << " nsample: " << nsample << std::endl;
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
        float* outData = (float*)out.data;

        int dim = xyz.dims;
        std::vector<int> inSize;
        for (int i = 0; i < dim; i++) {
            inSize.push_back(xyz.total(i, i + 1));
        }

        // TODO Implement the index points
    }
private:
    float radius;
    int nsample;
};


int main(int argc, char* argv[])
{
    //int testAxis = 2;
    //std::vector<float> testData;
    //for (int i = 0; i < 15; ++i) {
    //    testData.push_back(i + 10);
    //}
    //cv::Mat testMat(testData);
    //std::vector<int> testShape{ 1, 5, 3 };
    //testMat = testMat.reshape(0, testShape);
    //int dim = testMat.dims;
    //std::vector<int> inSize;
    //for (int i = 0; i < dim; i++) {
    //    inSize.push_back(testMat.total(i, i + 1));
    //}
    //
    //int n = inSize[3 - testAxis];  // testAxis = 1
    //std::vector<int> testRes;
    //for (int i = 0; i < n; ++i) {
    //    int maxId = 0;
    //    if (testAxis == 1) {
    //        float maxV = testMat.ptr<float>(0, 0)[i];  // T.ptr<float>(0, :)[i]
    //        for (int j = 1; j < inSize[testAxis]; ++j) {
    //            if (testMat.ptr<float>(0, j)[i] > maxV) {
    //                maxV = testMat.ptr<float>(0, j)[i];
    //                maxId = j;
    //            }
    //        }
    //        testRes.push_back(maxId);
    //    }
    //    else if (testAxis == 2) {
    //        float maxV = testMat.ptr<float>(0, i)[0];  // T.ptr<float>(0, i)[:]
    //        for (int j = 1; j < inSize[testAxis]; ++j) {
    //            if (testMat.ptr<float>(0, i)[j] > maxV) {
    //                maxV = testMat.ptr<float>(0, i)[j];
    //                maxId = j;
    //            }
    //        }
    //        testRes.push_back(maxId);
    //    }
    //}
    
    // load data
    std::string dataPath = "D:/Debug_dir/news_data/pcd_label_normal/bankou (1)_minCruv.pcd";  // argv[1]

    std::vector<std::vector<float> > points;                // N * 3
    if (!readFile(dataPath, points)) {
        return -1;
    }
    
    // preprocess data
    std::vector<int> ids;
    if (!PreProcess(points, ids)) {
        return -1;
    }

    cv::Mat datas(points.at(0).size(), points.size(), CV_32FC1);  // 3 * N
    for (int i = 0; i < datas.rows; ++i) {
        for (int j = 0; j < datas.cols; ++j) {
            datas.at<float>(i, j) = points.at(j).at(i);
        }
    }       

    std::vector<int> newShape{1, 1, datas.rows, datas.cols};
     datas = datas.reshape(0, newShape);    // 1 * 1 * 3 * N

    //// test
    //int size[3] = { 1, 2, 3 };
    //cv::Mat mat3D(3, size, CV_8UC1, cv::Scalar::all(0));
    
    /*float a = datas.ptr<float>(0, 0, 0)[0];
    float b = datas.ptr<float>(0, 0, 1)[0];
    float c = datas.ptr<float>(0, 0, 2)[0];*/

    // load model
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    
    const std::string modelPath = "E:/code/Server223/pointNet/inference_C++/test.onnx";
    //CV_DNN_REGISTER_LAYER_CLASS(ArgMax, ArgMaxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(fps, FPSLayer);
    CV_DNN_REGISTER_LAYER_CLASS(idx_pts, IndexPtsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(query_ball_pts, QueryBallPtsLayer);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    net.setInput(datas, "points");

    cv::Mat res = net.forward();
    int nRes = res.total();
    std::cout << " total item: " << nRes << std::endl;
    
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = endTime - startTime;
    std::cout << "all code took " << time_span.count() << " seconds." << std::endl;
                                                         
    
    return 0;
}