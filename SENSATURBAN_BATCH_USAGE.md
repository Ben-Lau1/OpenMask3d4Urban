# SensatUrban 批量处理使用指南

## 数据集结构要求

你的 SensatUrban 数据集应该组织成以下结构（与 `single_example` 相同）：

```
SENSATURBAN_SCANS_PATH/
├── scene1/
│   ├── pose/
│   │   ├── 0.txt
│   │   ├── 1.txt
│   │   └── ...
│   ├── color/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── depth/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── intrinsic/
│   │   └── intrinsic_color.txt
│   └── scene1.ply
├── scene2/
│   ├── pose/
│   ├── color/
│   ├── depth/
│   ├── intrinsic/
│   └── scene2.ply
└── ...
```

## 使用步骤

### 1. 修改脚本参数

编辑 `run_openmask3d_sensaturban_batch.sh`，设置以下参数：

```bash
# 数据集路径
SENSATURBAN_SCANS_PATH="/path/to/your/sensaturban/scans"

# 模型检查点
MASK_MODULE_CKPT_PATH="/path/to/scannet200_val.ckpt"
SAM_CKPT_PATH="/path/to/sam_vit_h_4b8939.pth"

# 数据格式参数（根据你的数据集调整）
IMG_EXTENSION=".jpg"  # 或 ".png"
DEPTH_EXTENSION=".png"
DEPTH_SCALE=1000
SCENE_INTRINSIC_RESOLUTION="[1024,2048]"  # 根据实际分辨率调整
```

### 2. 运行批量处理

```bash
bash run_openmask3d_sensaturban_batch.sh
```

脚本会自动：
1. 遍历所有场景目录
2. 对每个场景计算类别无关掩码
3. 批量计算所有场景的掩码特征
4. （可选）如果有 GT 标签，运行评估

### 3. 输出结构

处理完成后，输出目录结构如下：

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

## 注意事项

1. **场景名称匹配**：掩码文件名（如 `scene1_masks.pt`）中的场景名称应该与场景目录名（如 `scene1/`）匹配。

2. **点云文件命名**：点云文件可以是 `scene_name.ply` 或任何 `.ply` 文件，脚本会自动查找。

3. **如果某个场景处理失败**：脚本会跳过该场景并继续处理下一个，错误信息会打印在控制台。

4. **内存和显存**：如果处理大量场景时遇到内存问题，可以：
   - 设置 `OPTIMIZE_GPU_USAGE=true`（但会变慢）
   - 分批处理场景（修改脚本中的场景列表）

5. **评估**：如果有 GT 标签，需要设置 `SENSATURBAN_GT_DIR` 变量（在脚本中取消注释并设置路径）。

## 单独处理某个场景

如果你想单独测试某个场景，可以使用单场景脚本：

```bash
# 修改 run_openmask3d_single_scene.sh 中的 SCENE_DIR 参数
SCENE_DIR="/path/to/sensaturban/scans/scene1"
bash run_openmask3d_single_scene.sh
```

## 故障排查

1. **找不到场景目录**：检查 `SENSATURBAN_SCANS_PATH` 是否正确，确保场景目录直接在该路径下。

2. **掩码文件未生成**：检查点云文件是否存在，模型检查点路径是否正确。

3. **特征计算失败**：检查相机内参、位姿、图像、深度图路径是否正确，文件是否存在。

4. **评估失败**：确保 GT 标签文件格式正确，路径设置正确。

