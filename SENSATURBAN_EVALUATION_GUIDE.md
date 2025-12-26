# SensatUrban 数据集评估指南（使用预训练模型）

## 概述

本指南说明如何**在不训练 Mask3D 的情况下**，使用 ScanNet 预训练模型评估 SensatUrban 数据集。

## 评估流程

完整的评估流程包括 3 个步骤：

1. **生成掩码** - 使用预训练 Mask3D 模型生成类别无关掩码
2. **提取特征** - 使用 CLIP 提取每个掩码的特征
3. **运行评估** - 计算 mAP 和 mIoU 指标

## 前置要求

### 1. 数据准备

确保 SensatUrban 数据已组织成以下结构：

```
SENSATURBAN_SCANS_PATH/
├── scene1/
│   ├── pose/          # 相机位姿
│   ├── color/         # RGB 图像
│   ├── depth/         # 深度图
│   ├── intrinsic/     # 相机内参
│   └── scene1.ply     # 点云（包含标签！）
├── scene2/
│   └── ...
```

**重要**：PLY 文件必须包含 `class` 或 `label` 字段作为 GT 标签。

### 2. 下载预训练模型

下载以下模型（无需训练）：

- **Mask3D 模型**：`scannet200_val.ckpt`
  - 下载链接：[ScanNet200 模型](https://drive.google.com/file/d/1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B/view?usp=sharing)
  
- **SAM 模型**：`sam_vit_h_4b8939.pth`
  - 下载链接：[SAM ViT-H](https://drive.google.com/file/d/1WHi0hBi0iqMZfk8l3rDXLrW4lEEgHm_y/view?usp=sharing)

## 步骤 1: 配置脚本参数

编辑 `run_openmask3d_sensaturban_batch.sh`：

```bash
# 数据集路径
SENSATURBAN_SCANS_PATH="/path/to/your/sensaturban/scans"

# 模型路径（使用预训练模型，无需训练）
MASK_MODULE_CKPT_PATH="/path/to/scannet200_val.ckpt"
SAM_CKPT_PATH="/path/to/sam_vit_h_4b8939.pth"

# 数据格式参数
IMG_EXTENSION=".jpg"
DEPTH_EXTENSION=".png"
DEPTH_SCALE=1000
SCENE_INTRINSIC_RESOLUTION="[1024,1024]"  # 根据实际分辨率调整

# 内存优化参数（重要！）
data.voxel_size=0.1  # 对于大场景（2200万+点），建议 0.1-0.15
model.num_queries=100  # 可以减少到 80-100
```

## 步骤 2: 运行完整流程

运行批量处理脚本，它会自动执行所有步骤：

```bash
CUDA_VISIBLE_DEVICES=1 bash run_openmask3d_sensaturban_batch.sh
```

脚本会自动：
1. ✅ 对所有场景生成掩码
2. ✅ 提取所有场景的特征
3. ✅ 运行评估（如果 GT 目录存在）

## 步骤 3: 单独运行评估（可选）

如果掩码和特征已经生成，只想重新评估：

```bash
cd openmask3d

python evaluation/run_eval_sensaturban.py \
    --gt_dir=/path/to/SENSATURBAN_SCANS_PATH \
    --mask_pred_dir=/path/to/output/masks \
    --mask_features_dir=/path/to/output/mask_features \
    --clip_model='ViT-L/14@336px'
```

## 评估输出

评估会输出以下指标：

### 实例分割指标（mAP）
- **AP**: 平均精度（所有 IoU 阈值的平均）
- **AP@50**: IoU 阈值 0.5 的平均精度
- **AP@25**: IoU 阈值 0.25 的平均精度

### 语义分割指标（mIoU）
- **IoU**: 每个类别的 IoU
- **mIoU**: 所有类别的平均 IoU

### 输出示例

```
================================================================================
Class                 |         AP |      AP@50 |      AP@25 ||        IoU
--------------------------------------------------------------------------------
ground                |     0.xxxx |     0.xxxx |     0.xxxx ||     0.xxxx
vegetation            |     0.xxxx |     0.xxxx |     0.xxxx ||     0.xxxx
building              |     0.xxxx |     0.xxxx |     0.xxxx ||     0.xxxx
...
--------------------------------------------------------------------------------
AVERAGE               |     0.xxxx |     0.xxxx |     0.xxxx ||     0.xxxx
================================================================================
```

## GT 标签格式

评估脚本支持两种 GT 标签格式：

### 格式 1: 语义标签（Raw Label）
- PLY 文件中的 `class`/`label` 字段直接是类别 ID（0-12）
- 例如：`class = 0` 表示 `ground`

### 格式 2: 实例标签（Instance Label）
- PLY 文件中的标签值 >= 1000
- 格式：`instance_id = class_id * 1000 + instance_id`
- 例如：`label = 2001` 表示类别 2（building）的第 1 个实例

评估脚本会自动检测并处理这两种格式。

## 常见问题

### Q1: 评估时找不到 GT 文件

**问题**：`[Warning] GT file not found for scene xxx`

**解决**：
- 确保 `SENSATURBAN_GT_DIR` 指向包含 PLY 文件的目录
- 确保 PLY 文件名与场景名称匹配（如 `scene1.ply` 对应场景 `scene1`）
- 检查 PLY 文件是否包含标签字段

### Q2: 掩码和点云数量不匹配

**问题**：`ValueError: operands could not be broadcast together`

**解决**：
- **不要下采样点云**！这会导致索引不匹配
- 只通过 `voxel_size` 来减少内存占用
- 确保掩码生成和特征提取使用相同的点云文件

### Q3: 内存不足（OOM）

**解决**：
- 增加 `voxel_size`（如 0.1 → 0.15 或 0.2）
- 减少 `num_queries`（如 100 → 80）
- 设置 `OPTIMIZE_GPU_USAGE=true`（但会变慢）

### Q4: PLY 文件加载失败

**问题**：`vertex element not found`

**解决**：
- 检查 PLY 文件格式是否正确
- 确保 PLY 文件是二进制格式（binary_little_endian）
- 检查文件是否损坏

## 评估流程详解

### 1. 掩码生成阶段

```bash
python get_masks_single_scene.py \
    general.checkpoint=${MASK_MODULE_CKPT_PATH} \
    general.scene_path=${SCENE_PLY_PATH} \
    data.voxel_size=0.1 \
    ...
```

**输出**：`{scene_name}_masks.pt`
- 形状：`(num_points, num_masks)`
- 布尔掩码，表示每个点属于哪个实例

### 2. 特征提取阶段

```bash
python compute_features_sensaturban.py \
    data.masks.masks_path=${MASK_SAVE_DIR} \
    ...
```

**输出**：`{scene_name}_openmask3d_features.npy`
- 形状：`(num_masks, 768)`
- CLIP 特征向量

### 3. 评估阶段

```bash
python evaluation/run_eval_sensaturban.py \
    --gt_dir=${SENSATURBAN_SCANS_PATH} \
    --mask_pred_dir=${MASK_SAVE_DIR} \
    --mask_features_dir=${MASK_FEATURE_SAVE_DIR}
```

**过程**：
1. 加载掩码和特征
2. 使用 CLIP 文本查询为每个掩码分配类别
3. 从 PLY 文件加载 GT 标签
4. 匹配预测和 GT 实例（IoU 阈值）
5. 计算 mAP 和 mIoU

## 输出文件结构

```
output/
└── TIMESTAMP-sensaturban/
    ├── masks/
    │   ├── scene1_masks.pt
    │   ├── scene2_masks.pt
    │   └── ...
    ├── mask_features/
    │   ├── scene1_openmask3d_features.npy
    │   ├── scene2_openmask3d_features.npy
    │   └── ...
    └── hydra_outputs/
        ├── class_agnostic_mask_computation/
        └── mask_features_computation/
```

## 快速开始

1. **配置路径**：编辑 `run_openmask3d_sensaturban_batch.sh` 中的路径
2. **运行脚本**：`bash run_openmask3d_sensaturban_batch.sh`
3. **查看结果**：评估结果会打印在终端，包括每个类别的 AP 和 IoU

## 注意事项

1. **不需要训练**：直接使用 ScanNet 预训练模型即可
2. **不要下采样点云**：会导致掩码和点云索引不匹配
3. **体素化是关键**：通过 `voxel_size` 控制内存占用
4. **GT 标签在 PLY 中**：确保 PLY 文件包含 `class` 或 `label` 字段

## 总结

使用预训练模型评估 SensatUrban 的完整流程：

```
数据准备 → 配置参数 → 运行脚本 → 查看评估结果
```

无需训练，直接使用 ScanNet 预训练模型即可！




