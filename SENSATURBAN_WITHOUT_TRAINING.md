# 使用 ScanNet 预训练模型在 SensatUrban 上实验

## 为什么可以直接使用 ScanNet 模型？

OpenMask3D 的设计理念是**类别无关的实例分割**（class-agnostic instance segmentation），这意味着：

1. **掩码生成模块是类别无关的**：
   - Mask3D 模型学习的是如何分割 3D 场景中的实例，而不是识别特定类别
   - 它生成的是"物体实例"的掩码，不依赖于具体的类别标签
   - 因此可以跨数据集使用

2. **特征提取使用 CLIP**：
   - 特征提取部分使用 CLIP（开放词汇模型），不依赖于特定数据集的类别
   - CLIP 可以理解任意文本查询，包括 SensatUrban 的类别

3. **开放词汇设计**：
   - 整个系统设计为开放词汇，理论上可以处理任意数据集

## 直接使用 ScanNet 模型的步骤

### 1. 下载 ScanNet 预训练模型

根据 README，你需要下载：
- **Mask module checkpoint**: [ScanNet200 模型](https://drive.google.com/file/d/1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B/view?usp=sharing)
- **SAM checkpoint**: [SAM ViT-H 模型](https://drive.google.com/file/d/1WHi0hBi0iqMZfk8l3rDXLrW4lEEgHm_y/view?usp=sharing)

### 2. 修改脚本参数

编辑 `run_openmask3d_sensaturban_batch.sh`：

```bash
# 模型检查点路径
MASK_MODULE_CKPT_PATH="/path/to/scannet200_val.ckpt"  # 使用 ScanNet 模型
SAM_CKPT_PATH="/path/to/sam_vit_h_4b8939.pth"
```

### 3. 可能需要调整的参数

由于数据集不同，你可能需要调整以下参数以获得更好的效果：

#### 掩码生成参数（在脚本中）：
```bash
model.num_queries=120  # 可以尝试增加到 150 或更多，如果场景中有很多物体
general.dbscan_eps=0.95  # DBSCAN 聚类参数，可以尝试 0.9-1.0
```

#### 特征提取参数（在配置文件中）：
```yaml
openmask3d:
  top_k: 5  # 每个掩码使用的视角数，可以尝试 3-10
  frequency: 10  # 图像采样频率，如果图像很多可以增加（如 20）
  vis_threshold: 0.2  # 可见性阈值，可以尝试 0.15-0.25
```

### 4. 运行实验

```bash
bash run_openmask3d_sensaturban_batch.sh
```

## 预期效果

### 可能的情况：

1. **效果较好**：
   - 如果 SensatUrban 的场景结构与 ScanNet 相似（室内场景、点云密度等）
   - 掩码质量应该还可以接受
   - 可以通过调整参数进一步优化

2. **效果一般**：
   - 如果两个数据集差异较大（如室内 vs 室外、点云密度不同等）
   - 掩码可能不够精确，但应该还是能生成一些合理的实例分割

3. **需要微调**：
   - 如果效果不理想，可以考虑在 SensatUrban 上微调模型
   - 但这需要 SensatUrban 的训练数据和标注

## 优化建议

### 1. 参数调优

如果初始效果不理想，可以尝试：

```bash
# 在 run_openmask3d_sensaturban_batch.sh 中调整
model.num_queries=150  # 增加查询数量
general.dbscan_eps=0.9  # 调整聚类参数

# 在 openmask3d_sensaturban_eval.yaml 中调整
openmask3d:
  top_k: 7  # 增加使用的视角数
  frequency: 5  # 使用更多图像（降低采样频率）
  vis_threshold: 0.15  # 降低可见性阈值，包含更多点
```

### 2. 后处理优化

如果掩码质量不够好，可以考虑：
- 使用 DBSCAN 进一步过滤小掩码
- 调整置信度阈值
- 使用 NMS（非极大值抑制）去除重叠掩码

### 3. 数据预处理

确保你的数据格式正确：
- 点云坐标系统（z-up right-handed）
- 相机内参格式
- 深度图缩放因子

## 如果效果不理想怎么办？

### 选项 1: 参数调优（推荐先尝试）
- 调整上述参数
- 尝试不同的 DBSCAN 参数
- 调整图像采样频率

### 选项 2: 使用通用模型
- README 中提到有一个"任意场景"的模型：[链接](https://drive.google.com/file/d/1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y/view?usp=share_link)
- 这个模型可能泛化性更好

### 选项 3: 微调模型（需要训练数据）
- 如果有 SensatUrban 的训练集和标注
- 可以在 ScanNet 模型基础上微调
- 这需要修改训练脚本和配置

### 选项 4: 仅使用特征提取部分
- 如果掩码质量太差
- 可以考虑使用其他方法生成掩码（如传统分割方法）
- 然后只使用 OpenMask3D 的特征提取部分

## 实验建议

1. **先在小规模数据上测试**：
   - 选择 1-2 个场景先测试
   - 检查掩码质量和特征提取是否正常

2. **逐步调整参数**：
   - 从默认参数开始
   - 根据结果逐步调整
   - 记录每次调整的效果

3. **可视化检查**：
   - 设置 `SAVE_VISUALIZATIONS=true` 查看掩码可视化
   - 设置 `SAVE_CROPS=true` 查看提取的图像裁剪

4. **评估指标**：
   - 如果有 GT，使用评估脚本检查 mAP 和 mIoU
   - 如果没有 GT，人工检查几个场景的掩码质量

## 总结

**可以直接使用 ScanNet 模型进行实验**，因为：
- 掩码模块是类别无关的
- 特征提取使用开放词汇的 CLIP
- 系统设计为跨数据集使用

如果效果不理想，优先尝试参数调优，这是最简单有效的方法。

