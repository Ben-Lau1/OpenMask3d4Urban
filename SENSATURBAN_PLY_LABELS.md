# SensatUrban PLY 文件标签支持

## 概述

SensatUrban 数据集的标签直接存储在 PLY 文件中，而不是单独的 .txt 文件。我已经修改了代码以支持从 PLY 文件读取标签。

## PLY 文件格式

SensatUrban PLY 文件包含以下字段：
- `x, y, z`: 点云坐标
- `red, green, blue` 或 `r, g, b`: 颜色信息
- `class` 或 `label` 或 `l`: 语义标签

## 修改的文件

### 1. `openmask3d/utils_sensaturban.py` (新建)
- 提供了 `load_ply_xyzrgbl_fast()` 函数，专门用于读取 SensatUrban PLY 文件
- 支持从 PLY 文件读取坐标、颜色和标签

### 2. `openmask3d/class_agnostic_mask_computation/get_masks_single_scene.py`
- 修改了 `load_ply()` 函数，优先使用 SensatUrban 专用加载器
- 如果加载失败，会自动回退到 Open3D 默认加载器

### 3. `openmask3d/evaluation/eval_sensatUrban.py`
- 修改了 `assign_instances_for_scan()` 函数，支持从 PLY 文件读取 GT 标签
- 修改了 `evaluate()` 函数，自动检测并加载 PLY 格式的 GT 文件

### 4. `openmask3d/evaluation/run_eval_sensaturban.py`
- 更新了参数说明，支持 PLY 文件路径

## 使用方法

### 评估时使用 PLY 文件作为 GT

有两种方式：

#### 方式 1: GT 文件与场景数据在同一目录

如果你的场景数据组织如下：
```
SENSATURBAN_SCANS_PATH/
├── scene1/
│   ├── scene1.ply  # 包含标签
│   ├── pose/
│   ├── color/
│   └── ...
├── scene2/
│   ├── scene2.ply  # 包含标签
│   └── ...
```

评估时，`gt_dir` 应该指向包含 PLY 文件的目录：
```bash
python evaluation/run_eval_sensaturban.py \
    --gt_dir=/path/to/SENSATURBAN_SCANS_PATH \
    --mask_pred_dir=/path/to/masks \
    --mask_features_dir=/path/to/features
```

脚本会自动查找 `{scene_name}.ply` 文件。

#### 方式 2: GT 文件在单独目录

如果 GT PLY 文件在单独的目录：
```
GT_DIR/
├── scene1.ply
├── scene2.ply
└── ...
```

直接指定该目录：
```bash
python evaluation/run_eval_sensaturban.py \
    --gt_dir=/path/to/GT_DIR \
    --mask_pred_dir=/path/to/masks \
    --mask_features_dir=/path/to/features
```

## 标签格式说明

SensatUrban PLY 文件中的标签是**语义标签**（class id），不是实例标签。评估脚本会：

1. 如果标签值 < 1000：认为是语义标签（Raw Label 模式）
2. 如果标签值 >= 1000：认为是实例标签（Instance Label 模式，ID // 1000 = class id）

## 注意事项

1. **标签字段名称**：代码会自动检测 `class`、`label` 或 `l` 字段
2. **颜色字段**：代码会自动检测 `red/green/blue` 或 `r/g/b` 字段
3. **回退机制**：如果 PLY 加载失败，会尝试加载对应的 .txt 文件
4. **标签值范围**：确保标签值在 `VALID_CLASS_IDS` 定义的范围内

## 验证

运行评估前，可以先用单个场景测试：

```python
from openmask3d.utils_sensaturban import load_ply_xyzrgbl_fast

points, colors, labels = load_ply_xyzrgbl_fast("/path/to/scene.ply")
print(f"Points: {points.shape}")
print(f"Colors: {colors.shape}, range: [{colors.min():.2f}, {colors.max():.2f}]")
print(f"Labels: {labels.shape}, unique: {np.unique(labels)}")
```

## 故障排查

1. **找不到标签字段**：
   - 检查 PLY 文件是否包含 `class`、`label` 或 `l` 字段
   - 使用 `PlyData.read()` 查看文件结构

2. **标签值不匹配**：
   - 检查标签值是否在 `VALID_CLASS_IDS` 中
   - 查看 `sensatUrban_constants.py` 中的类别定义

3. **评估失败**：
   - 确保 PLY 文件路径正确
   - 检查场景名称是否匹配（掩码文件名中的场景名应该与 PLY 文件名匹配）

