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
        const float* inpData = (float*)inp.data;
        float* outData = (float*)out.data;
         
        int dim = inp.dims;
        std::vector<int> inSize;
        for (int i = 0; i < dim; i++) {
            inSize.push_back(inp.total(i, i + 1));
        }
        
        // Just implement axis = 1 or 2 for 3D Mat
        int batch = inSize[0];
        for (int b = 0; b < batch; ++b) {
            int n = inSize[3 - axis];
            for (int i = 0; i < n; ++i) {
                int maxId = 0;
                if (axis == 1) {
                    float maxV = inp.ptr<float>(0, 0)[i];  // T.ptr<float>(0, :)[i]
                    for (int j = 1; j < inSize[axis]; ++j) {
                        if (inp.ptr<float>(0, j)[i] > maxV) {
                            maxV = inp.ptr<float>(0, j)[i];
                            maxId = j;
                        }
                    }
                    outData[b * n + i] = float(maxId);
                }
                else if (axis == 2) {
                    float maxV = inp.ptr<float>(0, i)[0];  // T.ptr<float>(0, i)[:]
                    for (int j = 1; j < inSize[axis]; ++j) {
                        if (inp.ptr<float>(0, i)[j] > maxV) {
                            maxV = inp.ptr<float>(0, i)[j];
                            maxId = j;
                        }
                    }
                    outData[b * n + i] = float(maxId);
                }
            }
        }
        
    }
private:
    int axis;
};



int main(int argc, char* argv[])
{
    int testAxis = 2;
    std::vector<float> testData;
    for (int i = 0; i < 15; ++i) {
        testData.push_back(i + 10);
    }
    cv::Mat testMat(testData);
    std::vector<int> testShape{ 1, 5, 3 };
    testMat = testMat.reshape(0, testShape);

    int dim = testMat.dims;
    std::vector<int> inSize;
    for (int i = 0; i < dim; i++) {
        inSize.push_back(testMat.total(i, i + 1));
    }
    
    int n = inSize[3 - testAxis];  // testAxis = 1
    std::vector<int> testRes;
    for (int i = 0; i < n; ++i) {
        int maxId = 0;
        if (testAxis == 1) {
            float maxV = testMat.ptr<float>(0, 0)[i];  // T.ptr<float>(0, :)[i]
            for (int j = 1; j < inSize[testAxis]; ++j) {
                if (testMat.ptr<float>(0, j)[i] > maxV) {
                    maxV = testMat.ptr<float>(0, j)[i];
                    maxId = j;
                }
            }
            testRes.push_back(maxId);
        }
        else if (testAxis == 2) {
            float maxV = testMat.ptr<float>(0, i)[0];  // T.ptr<float>(0, i)[:]
            for (int j = 1; j < inSize[testAxis]; ++j) {
                if (testMat.ptr<float>(0, i)[j] > maxV) {
                    maxV = testMat.ptr<float>(0, i)[j];
                    maxId = j;
                }
            }
            testRes.push_back(maxId);
        }
    }


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
    
    const std::string modelPath = "E:/code/Server223/pointNet/inference_C++/testC++.onnx";
    CV_DNN_REGISTER_LAYER_CLASS(ArgMax, ArgMaxLayer);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    net.setInput(datas, "points");

    cv::Mat res = net.forward();
    int nRes = res.total();
    std::cout << "res: " << res << " total item: " << nRes << std::endl;
    
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = endTime - startTime;
    std::cout << "all code took " << time_span.count() << " seconds." << std::endl;
                                                         
    
    return 0;
}