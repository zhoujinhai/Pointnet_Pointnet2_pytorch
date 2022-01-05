### 模型降低pytorch版本流程
- 训练（pytorch==1.7.0）得到模型文件 代码见`train_partseg_tooth.py`
- 模型转换为torch==1.3.0版本（<font color="red">Linux gcc版本为4.8.5, 最高支持1.3.0</font>） 代码为`net2TorchscriptByPytorch_1_3_0.py`

### pytorch模型转onnx模型
- 不支持的操作在pytorch下以torch C++改写相关算子，然后生成对应的动态库 代码见`inference_onnx/ops.cpp`
- pytorch下加载动态库，并注册相关算子到torch.ops以及torch.onnx  代码为`net2onnx.py`
- pytorch模型中forward相应部分改写成自定义算子
- 导出onnx模型   

### 相关版本
- OpenCV => 4.5.3.36
- Operating System / Platform => Windows 64 Bit
- Compiler => Visual Studio 2019
- Python => 3.7.4
- Onnx => 1.10.2
- Onnxruntime => 1.9.0


