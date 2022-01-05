### pytorch模型转onnx
- 调整相关代码

  如：
  ```python
  mask = dist < distance
  distance[mask] = dist[mask]
  ```
  调整为
  ```python
  distance = torch.where(dist < distance, dist, distance)
  ```

- 自定义不支持的算子

  pytorch下有些算子不支持导出成onnx, 需要自定义该算子. 用torch C++改写后生成动态库(`ops.cpp`), 然后在pytorch中加载动态库,具体见`net2onnx.py`
  
- onnx调用自定义算子
   
  参考： https://github.com/onnx/tutorials/tree/master/PyTorchCustomOperator
  