import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= 路径配置 =================
# 输入：原始数据所在位置
SRC_2D_ROOT = Path("data/2d_revision_full_data")  # 包含 birmingham_block_0 文件夹
SRC_PLY_ROOT = Path("data/sensat_train")          # 包含 birmingham_block_0.ply
INTRINSIC_FILE = SRC_2D_ROOT / "intrinsics.txt"    # 公共内参文件

# 输出：你想要生成的最终目录
DST_ROOT = Path("data/SENSATURBAN_SCANS")

# 模式选择：
# "symlink" = 创建软链接 (推荐！速度快，几乎不占硬盘空间)
# "copy"    = 复制文件 (速度慢，占用双倍硬盘空间)
MODE = "symlink" 
# ===========================================

def create_formatted_dataset():
    if not SRC_2D_ROOT.exists():
        print(f"❌ 错误: 找不到源目录 {SRC_2D_ROOT}")
        return

    # 获取所有场景名称 (例如 birmingham_block_0, birmingham_block_1 ...)
    scene_names = [p.name for p in SRC_2D_ROOT.iterdir() if p.is_dir()]
    scene_names.sort()

    print(f"🔍 发现 {len(scene_names)} 个场景，准备处理...")
    print(f"📂 目标路径: {DST_ROOT}")
    print(f"⚙️  处理模式: {MODE} (软链接)")

    # 创建目标根目录
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for scene_id in tqdm(scene_names):
        # 1. 定义源路径
        src_scene_dir = SRC_2D_ROOT / scene_id
        src_ply_file = SRC_PLY_ROOT / f"{scene_id}.ply"
        
        # 2. 定义目标路径 (结构: scene_id/color, scene_id/pose ...)
        dst_scene_dir = DST_ROOT / scene_id
        dst_intrinsic_dir = dst_scene_dir / "intrinsic"
        
        # 创建目标文件夹结构
        dst_scene_dir.mkdir(exist_ok=True)
        dst_intrinsic_dir.mkdir(exist_ok=True)

        # -------------------------------------------------
        # (A) 处理 color, depth, pose 文件夹
        # -------------------------------------------------
        for subdir in ["color", "depth", "pose"]:
            src_sub = src_scene_dir / subdir
            dst_sub = dst_scene_dir / subdir
            
            # 如果目标已存在，先清理 (避免混合旧数据)
            if dst_sub.exists() or dst_sub.is_symlink():
                if dst_sub.is_symlink() or dst_sub.is_file():
                    dst_sub.unlink()
                else:
                    shutil.rmtree(dst_sub)

            if src_sub.exists():
                if MODE == "symlink":
                    # 创建软链接：dst -> src
                    os.symlink(src_sub.resolve(), dst_sub)
                else:
                    shutil.copytree(src_sub, dst_sub)

        # -------------------------------------------------
        # (B) 处理 Intrinsic (内参)
        # -------------------------------------------------
        # 目标: intrinsic/intrinsic_color.txt
        dst_intrinsic_file = dst_intrinsic_dir / "intrinsic_color.txt"
        
        if dst_intrinsic_file.exists() or dst_intrinsic_file.is_symlink():
            dst_intrinsic_file.unlink()

        if INTRINSIC_FILE.exists():
            if MODE == "symlink":
                os.symlink(INTRINSIC_FILE.resolve(), dst_intrinsic_file)
            else:
                shutil.copy(INTRINSIC_FILE, dst_intrinsic_file)
        else:
            print(f"⚠️ 警告: 公共内参文件缺失: {INTRINSIC_FILE}")

        # -------------------------------------------------
        # (C) 处理 .ply 点云文件
        # -------------------------------------------------
        # 目标: scene_id/scene_id.ply
        dst_ply_file = dst_scene_dir / f"{scene_id}.ply"

        if dst_ply_file.exists() or dst_ply_file.is_symlink():
            dst_ply_file.unlink()

        if src_ply_file.exists():
            if MODE == "symlink":
                os.symlink(src_ply_file.resolve(), dst_ply_file)
            else:
                shutil.copy(src_ply_file, dst_ply_file)
        else:
            print(f"⚠️ 警告: 场景 {scene_id} 对应的点云文件不存在")

    print("\n✅ 所有数据整理完毕！")

if __name__ == "__main__":
    create_formatted_dataset()