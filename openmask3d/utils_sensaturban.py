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
    
    注意：此函数返回完整的点云，不下采样。内存优化应通过体素化（voxel_size）实现。
    """
    # 使用文件句柄读取，避免某些平台上出现奇怪的对象类型
    with open(ply_path, "rb") as f:
        ply = PlyData.read(f)
    
    # 检查 vertex 元素是否存在
    # ply.elements 可能是字典或列表，需要兼容处理
    if isinstance(ply.elements, dict):
        element_names = list(ply.elements.keys())
        if "vertex" in ply.elements:
            v = ply["vertex"].data
        else:
            raise ValueError(f"vertex element not found in {ply_path}. Available elements: {element_names}")
    else:
        # ply.elements 是列表
        element_names = [elem.name for elem in ply.elements]
        if "vertex" not in element_names:
            raise ValueError(f"vertex element not found in {ply_path}. Available elements: {element_names}")
        
        # 尝试通过字典方式访问
        try:
            v = ply["vertex"].data
        except (KeyError, TypeError):
            # 如果字典访问失败，通过列表查找
            for elem in ply.elements:
                if elem.name == "vertex":
                    v = elem.data
                    break
            else:
                raise ValueError(f"Cannot access vertex data from {ply_path}")
    logging.info(f"PLY vertex fields: {v.dtype.names}")

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

