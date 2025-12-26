#!/bin/bash
export OMP_NUM_THREADS=4  # speeds up MinkowskiEngine
# 自动获取当前脚本所在的绝对路径
SCRIPT_DIR=$(cd $(dirname $0); pwd)

# 将核心代码目录和嵌套的 datasets 目录都加入 Python 搜索路径
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/openmask3d/class_agnostic_mask_computation
set -e

# OPENMASK3D SENSATURBAN BATCH PROCESSING SCRIPT
# This script processes multiple SensatUrban scenes in batch
# Each scene should have the same structure as single_example:
#   scene_name/
#     ├── pose/
#     ├── color/
#     ├── depth/
#     ├── intrinsic/
#     └── scene_name.ply

# --------
# NOTE: SET THESE PARAMETERS!
# 数据集根目录（包含所有场景文件夹）
SENSATURBAN_SCANS_PATH="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/openmask3d_old/openmask3d/data/SENSATURBAN_SCANS"  # 修改为你的数据集路径
# 模型检查点路径
# 注意：可以直接使用 ScanNet 预训练模型，因为掩码模块是类别无关的
# 下载链接见 README.md 或 SENSATURBAN_WITHOUT_TRAINING.md
MASK_MODULE_CKPT_PATH="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/openmask3d_old/openmask3d/scannet200_val.ckpt"  # ScanNet 预训练模型即可
SAM_CKPT_PATH="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/openmask3d_old/openmask3d/sam_vit_h_4b8939.pth"
# 输出目录
EXPERIMENT_NAME="sensaturban"
OUTPUT_DIRECTORY="$(pwd)/output"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
MASK_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks"
MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
# 其他参数
SAVE_VISUALIZATIONS=false  # if set to true, saves pyviz3d visualizations
SAVE_CROPS=false
OPTIMIZE_GPU_USAGE=false
# 数据格式参数（根据你的数据集调整）
IMG_EXTENSION=".jpg"  # 或 ".png"
DEPTH_EXTENSION=".png"  # 或 ".jpg"
DEPTH_SCALE=1000  # 根据数据集调整（ScanNet=1000, Replica=6553.5）
SCENE_INTRINSIC_RESOLUTION="[1024,2048]"  # 根据你的内参分辨率调整

# 创建输出目录
mkdir -p "${MASK_SAVE_DIR}"
mkdir -p "${MASK_FEATURE_SAVE_DIR}"

cd openmask3d

# 获取所有场景目录
SCENE_DIRS=($(find "${SENSATURBAN_SCANS_PATH}" -mindepth 1 -maxdepth 1 -type d | sort))

echo "[INFO] Found ${#SCENE_DIRS[@]} scenes to process"
echo "[INFO] Output directory: ${OUTPUT_FOLDER_DIRECTORY}"

# 步骤1: 批量计算类别无关掩码
echo "=========================================="
echo "[INFO] Step 1: Computing class agnostic masks for all scenes..."
echo "=========================================="

SCENE_COUNT=0
for SCENE_DIR in "${SCENE_DIRS[@]}"; do
    SCENE_NAME=$(basename "${SCENE_DIR}")
    SCENE_PLY_PATH=$(find "${SCENE_DIR}" -maxdepth 1 -name "*.ply" | head -1)
    
    # 检查必要的文件是否存在
    if [ ! -f "${SCENE_PLY_PATH}" ]; then
        echo "[WARNING] No .ply file found in ${SCENE_DIR}, skipping scene ${SCENE_NAME}"
        continue
    fi
    
    SCENE_COUNT=$((SCENE_COUNT + 1))
    echo "[INFO] Processing scene ${SCENE_COUNT}/${#SCENE_DIRS[@]}: ${SCENE_NAME}"
    
    # 计算掩码
    python class_agnostic_mask_computation/get_masks_single_scene.py \
    general.experiment_name=${EXPERIMENT_NAME} \
    general.checkpoint=${MASK_MODULE_CKPT_PATH} \
    general.train_mode=false \
    data.test_mode=test \
    model.num_queries=120 \
    general.use_dbscan=true \
    general.dbscan_eps=0.95 \
    general.save_visualizations=${SAVE_VISUALIZATIONS} \
    general.scene_path=${SCENE_PLY_PATH} \
    general.mask_save_dir="${MASK_SAVE_DIR}" \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation/${SCENE_NAME}"
    
    # 检查掩码文件是否生成
    MASK_FILE_BASE=$(basename "${SCENE_PLY_PATH}" .ply)
    MASK_FILE_NAME="${MASK_FILE_BASE}_masks.pt"
    SCENE_MASK_PATH="${MASK_SAVE_DIR}/${MASK_FILE_NAME}"
    
    if [ -f "${SCENE_MASK_PATH}" ]; then
        echo "[INFO] Masks saved for ${SCENE_NAME}: ${SCENE_MASK_PATH}"
    else
        echo "[WARNING] Mask file not found for ${SCENE_NAME}, expected: ${SCENE_MASK_PATH}"
    fi
done

echo "[INFO] Mask computation completed for ${SCENE_COUNT} scenes!"

# 步骤2: 批量计算掩码特征
echo "=========================================="
echo "[INFO] Step 2: Computing mask features for all scenes..."
echo "=========================================="

# 使用批量特征计算脚本（基于掩码文件自动查找场景）
python compute_features_sensaturban.py \
data.scans_path=${SENSATURBAN_SCANS_PATH} \
data.masks.masks_path=${MASK_SAVE_DIR} \
data.masks.masks_suffix='*_masks.pt' \
data.camera.poses_path='pose' \
data.camera.intrinsic_path='intrinsic/intrinsic_color.txt' \
data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
data.depths.depths_path='depth' \
data.depths.depth_scale=${DEPTH_SCALE} \
data.depths.depths_ext=${DEPTH_EXTENSION} \
data.images.images_path='color' \
data.images.images_ext=${IMG_EXTENSION} \
output.output_directory=${MASK_FEATURE_SAVE_DIR} \
output.experiment_name=${EXPERIMENT_NAME} \
output.save_crops=${SAVE_CROPS} \
external.sam_checkpoint=${SAM_CKPT_PATH} \
external.sam_model_type='vit_h' \
external.clip_model='ViT-L/14@336px' \
openmask3d.top_k=5 \
openmask3d.multi_level_expansion_ratio=0.1 \
openmask3d.num_of_levels=3 \
openmask3d.vis_threshold=0.2 \
openmask3d.frequency=10 \
openmask3d.num_random_rounds=10 \
openmask3d.num_selected_points=5 \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"

echo "[INFO] Feature computation completed!"

# 步骤3: 评估（如果有 GT 标签）
if [ -n "${SENSATURBAN_GT_DIR}" ] && [ -d "${SENSATURBAN_GT_DIR}" ]; then
    echo "=========================================="
    echo "[INFO] Step 3: Running evaluation..."
    echo "=========================================="
    
    python evaluation/run_eval_sensaturban.py \
    --gt_dir=${SENSATURBAN_GT_DIR} \
    --mask_pred_dir=${MASK_SAVE_DIR} \
    --mask_features_dir=${MASK_FEATURE_SAVE_DIR}
    
    echo "[INFO] Evaluation completed!"
else
    echo "[INFO] Skipping evaluation (GT directory not set or not found)"
fi

echo "=========================================="
echo "[INFO] All processing completed!"
echo "[INFO] Results saved to: ${OUTPUT_FOLDER_DIRECTORY}"
echo "=========================================="

