### 模型降低pytorch版本流程
- 训练（pytorch==1.7.0）得到模型文件 代码见`train_partseg_tooth.py`
- 模型在centos7 gcc=4.8.5环境下运行, 需要切换torch==1.3.0版本（<font color="red">Linux gcc版本为4.8.5, 最高支持1.3.0</font>） 
切换版本后重新转换 代码为`net2Torchscript.py`

### pytorch模型转torch script模型
```bash
python net2Torchscript.py
```

### 调用代码
见`test.cpp`