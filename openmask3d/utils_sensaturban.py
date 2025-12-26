"""
SensatUrban 数据集相关的工具函数
"""
import numpy as np
from plyfile import PlyData
import logging

def load_ply_xyzrgbl_fast(ply_path):
    """读取 SensatUrban 单个 .ply, 假定顶点字段为 xyzrgbl.
    
    Args:
        ply_path: PLY 文件路径
        
    Returns:
        points: (N, 3) 点云坐标
        colors: (N, 3) 颜色，范围 [-1, 1]
        labels: (N,) 标签，如果没有标签则全为0
    """
    ply = PlyData.read(ply_path)
    logging.info(f"PLY vertex fields: {ply['vertex'].dtype.names}")
    
    # 通常 SensatUrban 顶点元素名为 "vertex"
    v = ply["vertex"].data

    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # 读取颜色
    if 'red' in v.dtype.names:
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
    elif 'r' in v.dtype.names:
        colors = np.stack([v["r"], v["g"], v["b"]], axis=1).astype(np.float32)
    else:
        logging.warning(f"No color field found in {ply_path}, using zeros")
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    
    # 读取标签
    if 'class' in v.dtype.names:
        labels = v['class'].astype(np.int32)
    elif 'label' in v.dtype.names:
        labels = v['label'].astype(np.int32)
    elif 'l' in v.dtype.names:
        labels = v['l'].astype(np.int32)
    else:
        logging.warning(f"No label field found in {ply_path}, using zeros")
        labels = np.zeros((points.shape[0],), dtype=np.int32)
    
    # 颜色归一化到 [-1, 1]
    colors = colors / 127.5 - 1.0
    
    return points, colors, labels


def load_labels_from_ply(ply_path):
    """从 PLY 文件加载标签（用于评估）
    
    Args:
        ply_path: PLY 文件路径
        
    Returns:
        labels: (N,) 标签数组
    """
    _, _, labels = load_ply_xyzrgbl_fast(ply_path)
    return labels

