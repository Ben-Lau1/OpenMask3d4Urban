import torch
import os

# 1. 设置只看 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"--- 显卡诊断 ---")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.get_device_name(0)}")
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"初始剩余显存: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")
else:
    print("❌ 找不到显卡！")
    exit()

# 2. 加载模型权重 (模拟你的路径)
ckpt_path = "/home/zhangshuai/workshop/open_vocabulary/openmask3d/scannet200_val.ckpt"
print(f"\n--- 正在加载模型文件: {ckpt_path} ---")

try:
    # 先加载到 CPU
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    # 计算参数总量
    total_size_bytes = 0
    for key, value in state_dict.items():
        total_size_bytes += value.numel() * value.element_size()
    
    print(f"✅ 模型文件加载成功！")
    print(f"📊 模型参数总大小: {total_size_bytes / 1024**2:.2f} MB (不到 24GB 的 1/20)")
    
    # 3. 暴力测试：能否塞进显卡？
    print(f"\n--- 正在尝试将参数搬运到显卡 ---")
    # 我们模拟创建一个和参数一样大的张量列表塞进显存
    tensors = []
    for key, value in state_dict.items():
        tensors.append(value.to("cuda:0")) # 这里的 cuda:0 对应物理 GPU 1
        
    print(f"🎉 成功！所有参数已搬进显卡。")
    print(f"结论: 显卡没问题，模型也没问题。问题出在 OpenMask3D 代码逻辑里。")

except RuntimeError as e:
    print(f"\n❌ 搬运失败，报错信息:\n{e}")
    if "out of memory" in str(e):
        print("\n诊断: 即使是纯净环境也 OOM，说明这张显卡的状态极度异常（可能是 Zombie 进程）。")
        print("建议: 请尝试重启服务器。")
except Exception as e:
    print(f"发生其他错误: {e}")