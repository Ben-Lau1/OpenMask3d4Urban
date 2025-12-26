import logging
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer
import open3d as o3d
import numpy as np
import torch
import time
import pdb

# 尝试导入 SensatUrban 专用加载函数
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from openmask3d.utils_sensaturban import load_ply_xyzrgbl_fast
    USE_SENSATURBAN_LOADER = True
except ImportError:
    USE_SENSATURBAN_LOADER = False
    logging.warning("SensatUrban loader not found, using default Open3D loader")

def get_parameters(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    #loggers = []

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    #logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, None #loggers


def load_ply(filepath):
    """加载 PLY 文件，支持 SensatUrban 格式（包含标签）"""
    if USE_SENSATURBAN_LOADER:
        try:
            # 尝试使用 SensatUrban 专用加载器
            coords, colors, labels = load_ply_xyzrgbl_fast(filepath)
            # 计算法线
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            # 将颜色从 [-1, 1] 转换回 [0, 1] 用于后续处理
            colors = (colors + 1.0) / 2.0
            return coords, colors, normals
        except Exception as e:
            logging.warning(f"Failed to use SensatUrban loader: {e}, falling back to Open3D")
    
    # 默认使用 Open3D 加载
    pcd = o3d.io.read_point_cloud(filepath)
    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals

def process_file(filepath):
    coords, colors, normals = load_ply(filepath)
    raw_coordinates = coords.copy()
    raw_colors = (colors*255).astype(np.uint8)
    raw_normals = normals

    features = colors
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    filename = filepath.split("/")[-1][:-4]
    return [[coords, features, [], filename, raw_colors, raw_normals, raw_coordinates, 0]] # 2: original_labels, 3: none
    # coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], raw_color, raw_normals, raw_coordinates, idx

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def get_class_agnostic_masks(cfg: DictConfig):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    c_fn = hydra.utils.instantiate(cfg.data.test_collation) #(model.config.data.test_collation)

    input_batch = process_file(cfg.general.scene_path)
    batch = c_fn(input_batch)

    # === 自动修复: 强力清理显存 ===
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Debug] 显存清理完毕，当前占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    # ============================
    model.to(device)
    model.eval()

    start = time.time()
    with torch.no_grad():
        res_dict = model.get_masks_single_scene(batch)
    end = time.time()
    print("Time elapsed: ", end - start)

@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks(cfg)

if __name__ == "__main__":
    main()
