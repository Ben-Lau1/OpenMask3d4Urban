import hydra
from omegaconf import DictConfig
import numpy as np
from openmask3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from openmask3d.utils import get_free_gpu, create_out_folder
from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor
import torch
import os
from glob import glob

# TIP: add version_base=None to the arguments if you encounter some error  
@hydra.main(config_path="configs", config_name="openmask3d_sensaturban_eval")
def main(ctx: DictConfig):
    device = "cpu"
    device = get_free_gpu(7000) if torch.cuda.is_available() else device
    print(f"[INFO] Using device: {device}")
    out_folder = ctx.output.output_directory
    os.chdir(hydra.utils.get_original_cwd())
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"[INFO] Saving feature results to {out_folder}")
    
    # 获取所有掩码文件
    masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_path, ctx.data.masks.masks_suffix)))
    
    for masks_path in masks_paths:
        # 从掩码文件名中提取场景名称
        # 假设掩码文件名格式为: scene_name_masks.pt 或 scene_name.ply_masks.pt
        mask_filename = os.path.basename(masks_path)
        scene_name = mask_filename.replace('_masks.pt', '').replace('masks.pt', '')
        
        # 如果掩码文件名包含 .ply，需要去掉
        if scene_name.endswith('.ply'):
            scene_name = scene_name[:-4]
        
        # 构建场景路径
        scene_path = os.path.join(ctx.data.scans_path, scene_name)
        
        # 如果场景路径不存在，尝试直接使用场景名称作为目录名
        if not os.path.exists(scene_path):
            # 尝试查找匹配的场景目录（可能场景名称不完全匹配）
            possible_dirs = [d for d in os.listdir(ctx.data.scans_path) 
                          if os.path.isdir(os.path.join(ctx.data.scans_path, d)) 
                          and scene_name in d]
            if len(possible_dirs) > 0:
                scene_path = os.path.join(ctx.data.scans_path, possible_dirs[0])
                scene_name = possible_dirs[0]
        
        poses_path = os.path.join(scene_path, ctx.data.camera.poses_path)
        
        # 查找点云文件
        point_cloud_pattern = os.path.join(scene_path, '*.ply')
        point_cloud_files = glob(point_cloud_pattern)
        if len(point_cloud_files) == 0:
            print(f"[WARNING] No point cloud found for scene {scene_name} in {scene_path}, skipping...")
            continue
        point_cloud_path = point_cloud_files[0]
        
        intrinsic_path = os.path.join(scene_path, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(scene_path, ctx.data.images.images_path)
        depths_path = os.path.join(scene_path, ctx.data.depths.depths_path)
        
        # 检查路径是否存在
        if not os.path.exists(poses_path):
            print(f"[WARNING] Poses path not found: {poses_path}, skipping scene {scene_name}")
            continue
        if not os.path.exists(intrinsic_path):
            print(f"[WARNING] Intrinsic path not found: {intrinsic_path}, skipping scene {scene_name}")
            continue
        if not os.path.exists(images_path):
            print(f"[WARNING] Images path not found: {images_path}, skipping scene {scene_name}")
            continue
        if not os.path.exists(depths_path):
            print(f"[WARNING] Depths path not found: {depths_path}, skipping scene {scene_name}")
            continue
        
        print(f"[INFO] Processing scene: {scene_name}")
        
        # 1. Load the masks
        masks = InstanceMasks3D(masks_path) 

        # 2. Load the images
        indices = np.arange(0, get_number_of_images(poses_path), step = ctx.openmask3d.frequency)
        images = Images(images_path=images_path, 
                        extension=ctx.data.images.images_ext, 
                        indices=indices)

        # 3. Load the pointcloud
        pointcloud = PointCloud(point_cloud_path)

        # 4. Load the camera configurations
        camera = Camera(intrinsic_path=intrinsic_path, 
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                        poses_path=poses_path, 
                        depths_path=depths_path, 
                        extension_depth=ctx.data.depths.depths_ext, 
                        depth_scale=ctx.data.depths.depth_scale)

        # 5. Run extractor
        features_extractor = FeaturesExtractor(camera=camera, 
                                                clip_model=ctx.external.clip_model, 
                                                images=images, 
                                                masks=masks,
                                                pointcloud=pointcloud, 
                                                sam_model_type=ctx.external.sam_model_type,
                                                sam_checkpoint=ctx.external.sam_checkpoint,
                                                vis_threshold=ctx.openmask3d.vis_threshold,
                                                device=device)

        features = features_extractor.extract_features(topk=ctx.openmask3d.top_k, 
                                                        multi_level_expansion_ratio = ctx.openmask3d.multi_level_expansion_ratio,
                                                        num_levels=ctx.openmask3d.num_of_levels, 
                                                        num_random_rounds=ctx.openmask3d.num_random_rounds,
                                                        num_selected_points=ctx.openmask3d.num_selected_points,
                                                        save_crops=ctx.output.save_crops,
                                                        out_folder=out_folder,
                                                        optimize_gpu_usage=ctx.gpu.optimize_gpu_usage)
        
        # 6. Save features
        filename = f"{scene_name}_openmask3d_features.npy"
        output_path = os.path.join(out_folder, filename)
        np.save(output_path, features)
        print(f"[INFO] Mask features for scene {scene_name} saved to {output_path}.")
    
    
    
if __name__ == "__main__":
    main()


