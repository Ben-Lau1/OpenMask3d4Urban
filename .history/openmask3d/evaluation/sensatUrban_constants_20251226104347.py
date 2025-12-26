# SensatUrban 数据集的常量定义
# 包含 13 个语义类别
# ID 范围: 0 - 12

# 1. 有效类别 ID (必须与你的 GT .txt 文件中的 ID 一致)
VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# 2. 对应的英文类别标签 (用于生成 CLIP 查询文本)
CLASS_LABELS = (
    'ground',            # 0
    'vegetation',        # 1
    'building',          # 2
    'wall',              # 3
    'bridge',            # 4
    'parking',           # 5
    'rail',              # 6
    'traffic road',      # 7
    'street furniture',  # 8
    'car',               # 9
    'footpath',          # 10
    'bike',              # 11
    'water'              # 12
)

# 3. 颜色映射 (ID -> RGB Tuple)，用于可视化调试
# 颜色取值参考了常见的语义分割配色方案
COLOR_MAP = {
    0: (85, 107, 47),    # ground (olive green)
    1: (0, 255, 0),      # vegetation (lime green)
    2: (255, 165, 0),    # building (orange)
    3: (47, 79, 79),     # wall (dark slate gray)
    4: (123, 104, 238),  # bridge (medium slate blue)
    5: (128, 128, 128),  # parking (gray)
    6: (255, 0, 255),    # rail (magenta)
    7: (70, 130, 180),   # traffic road (steel blue)
    8: (220, 20, 60),    # street furniture (crimson)
    9: (0, 0, 255),      # car (blue)
    10: (139, 69, 19),   # footpath (saddle brown)
    11: (0, 255, 255),   # bike (cyan)
    12: (0, 191, 255),   # water (deep sky blue)
}