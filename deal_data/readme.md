### 数据集制作

#### 手动标注
借助[`Semantic Segmentation Editor`](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor) 工具
具体安装及操作可参考：
https://www.cnblogs.com/xiaxuexiaoab/p/15250486.html

### 根据牙模和牙龈线自动生成
#### deal_file.py
**功能**
- 从各文件夹中挑选出stl模型及牙龈线文件pts

#### deal_data_Class.py
**功能**
- 根据obj模型及牙龈线pts文件生成pcd文件

#### Mesh_Simplifier.py
**功能**
- 借助hgapi将stl直接下采样生成obj牙模