### 流程
#### 1. 训练模型
- 准备好训练数据, 如pcd文件
- 数据加载见`data_utils/ToothPcdDataLoader.py`
- 训练模型
```bash
python train_partset_tooth.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg_gumline
# python train_semseg_tooth.py --model pointnet2_sem_seg_msg --log_dir pointnet2_seg_msg_tooth
```
- 模型保存至`log/part_seg/pointnet2_part_seg_msg_gumline`

#### 2. 模型参数调整
运行`netChangeParams.py`
```bash
python netChangeParams.py
```

#### 3. 模型格式转换
- 转为onnx
运行`net2onnx.py`
```bash
python net2onnx.py
```

- 转为torchscript
运行`net2Torchscript.py`
```bash
python net2Torchscript.py
```

#### 4. C++推理
- OpenCV调用onnx模型
见inference_cv文件夹

- libtorch调用script模型
见inference_libtorch文件夹
