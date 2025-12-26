# 内存优化指南 - SensatUrban 大数据集

## 问题诊断

你遇到的问题是：
1. **进程被杀死（Killed）**：通常是内存不足（OOM）导致的
2. **点云规模**：SensatUrban 场景有 **2200万+ 个点**，远超 ScanNet（通常几百万点）

## 解决方案

### ⚠️ 重要：不要下采样点云！

**原因**：下采样点云会导致掩码和点云索引不匹配！
- 掩码生成阶段：如果下采样点云，掩码的维度是 `(下采样后的点数, num_masks)`
- 特征提取阶段：使用原始点云，点云维度是 `(原始点数,)`
- 结果：`masks.masks[:,j] * visible_points_view[i]` 会因维度不匹配而失败

### ✅ 方案 1: 增加体素化尺寸（唯一推荐方案）

**原理**：体素化会将点云转换为稀疏体素表示，体素尺寸越大，体素数越少，内存占用越低。
- 体素化是**确定性的**，不会破坏点云和掩码的对应关系
- 体素化后的点数会远少于原始点数

修改方式：

**方法1：修改配置文件**
```yaml
# openmask3d/class_agnostic_mask_computation/conf/data/indoor.yaml
voxel_size: 0.1  # 从 0.5 改为 0.1 或更大（如 0.15, 0.2）
```

**方法2：命令行参数覆盖（推荐）**
```bash
python class_agnostic_mask_computation/get_masks_single_scene.py \
    data.voxel_size=0.1 \
    ...
```

**建议值**：
- ScanNet（室内，几百万点）: 0.02
- SensatUrban（室外，2200万点）: **0.1 - 0.2**
  - 0.1: 中等内存占用，较好的精度
  - 0.15: 较低内存占用，精度略降
  - 0.2: 最低内存占用，精度明显降低

### ❌ 方案 2: 点云下采样（已移除，不推荐）

**已移除下采样功能**，因为会导致掩码和点云索引不匹配。

如果需要减少内存占用，**只使用体素化**。

### 方案 3: 增加系统内存/交换空间

如果硬件允许：
- 增加系统 RAM
- 增加 swap 空间
- 使用更大内存的 GPU

### 方案 4: 分批处理大场景

将大场景分割成多个小块，分别处理后再合并。

### 方案 5: 调整模型参数

减少内存占用的参数：

```bash
model.num_queries=100  # 减少查询数量（从 120 降到 100）
data.batch_size=1      # 确保 batch_size=1
```

## 推荐配置

对于 SensatUrban（2200万点），推荐配置：

```bash
# 在 run_openmask3d_sensaturban_batch.sh 中
data.voxel_size=0.1  # 关键：增加体素尺寸以减少内存
model.num_queries=100  # 减少查询数
general.use_dbscan=true
general.dbscan_eps=0.95
```

**不要下采样点云**，只通过体素化来减少内存占用。

## 验证内存使用

处理前检查点云大小：

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("path/to/scene.ply")
print(f"Points: {len(pcd.points)}")
```

如果超过 1000 万点，建议：
1. **只增加 `voxel_size`** 到 0.1-0.2（不要下采样点云）
2. 如果还是 OOM，进一步增加 `voxel_size` 到 0.2-0.3
3. 或者减少 `num_queries` 到 80-100

## 体素化 vs 下采样

| 方法 | 优点 | 缺点 | 推荐 |
|------|------|------|------|
| **体素化** | 确定性、保持对应关系、内存友好 | 精度略降 | ✅ **推荐** |
| **下采样** | 简单 | 破坏对应关系、导致错误 | ❌ **不推荐** |

**体素化是唯一安全的内存优化方法**。

## 监控内存使用

运行时监控内存：

```bash
# 在另一个终端运行
watch -n 1 free -h
# 或
nvidia-smi -l 1  # GPU 内存
```

如果看到内存持续增长直到被杀死，说明需要优化。

