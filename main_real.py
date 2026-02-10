# main_real.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import ptych
import os
import json

from ptych import solve_inverse
from utils import (
    get_default_device,
    load_real_captures,
    calculate_k_vectors_from_positions,
    create_circular_pupil,
    visualize_kspace_and_captures
)


# --- 1. 用户配置 (User Configuration) ---
# ----------------------------------------------------

# 定义配置文件路径
CONFIG_JSON_PATH = r"D:\FPM_Dataset\OnTest\param.json"

if not os.path.exists(CONFIG_JSON_PATH):
    raise FileNotFoundError(f"找不到配置文件: {CONFIG_JSON_PATH}")

with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# A. 从 JSON 中读取系统物理参数
NA_OBJECTIVE = config_data.get('NA_OBJECTIVE', 0.5)          # 物镜 NA
WAVELENGTH_NM = config_data.get('WAVELENGTH_NM', 465)        # LED 波长 (nm)
MAGNIFICATION = config_data.get('MAGNIFICATION', 10.0)       # 物镜放大倍率
CAMERA_PIXEL_SIZE_UM = config_data.get('CAMERA_PIXEL_SIZE_UM', 3.45) # 相机像素尺寸 (um)

#print(f"--- 系统参数已加载 ---")
print(f"NA: {NA_OBJECTIVE}\nWavelength: {WAVELENGTH_NM}nm\nMag: {MAGNIFICATION}x\nAMERA_PIXEL_SIZE:{CAMERA_PIXEL_SIZE_UM}\n")

# B. 数据和重建参数
CAPTURES_PATH = "D:\FPM_Dataset\OnTest\TIF" # 你存放真实图像的文件夹
LED_POSITIONS_FILE = "led_positions.csv" # 你的LED位置文件
CENTER_LED_INDEX = 1        # 对应中心照明的LED的索引号 (从1开始)
DOWNSAMPLE_FACTOR = 2       # 生成图片分辨率倍数


# --- 2. 初始化和设备设置 ---
# ----------------------------------------------------
pytorch_device = get_default_device()
torch.set_default_device(pytorch_device)
print(f"Running on device: {pytorch_device}")


# --- 3. 加载并预处理真实数据 ---
# ----------------------------------------------------
# load_real_captures 现在返回两个值
captures, loaded_led_indices = load_real_captures(CAPTURES_PATH, file_pattern="*.tif")

captures = captures.to(pytorch_device)

# 预处理步骤保持不变
captures = captures / captures.mean(dim=(-1, -2), keepdim=True)


# --- 4. 计算和创建初始猜测 ---
# ----------------------------------------------------
# A. 确定重建尺寸
capture_height, capture_width = captures.shape[-2:]
output_size = capture_height * DOWNSAMPLE_FACTOR # 最终重建图像尺寸
print(f"Capture size: {capture_height}x{capture_width}, Output size: {output_size}x{output_size}")

# B. 计算 k-vectors 初始估计
recon_pixel_size_m = (CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION) * DOWNSAMPLE_FACTOR
# 首先，计算出你的 .csv 文件中所有LED的k-vectors
all_kx, all_ky = calculate_k_vectors_from_positions(
    LED_POSITIONS_FILE,
    WAVELENGTH_NM,
    MAGNIFICATION,
    CAMERA_PIXEL_SIZE_UM,
    recon_pixel_size_m,
    center_led_index=CENTER_LED_INDEX
)

all_kx = all_kx.to(pytorch_device)
all_ky = all_ky.to(pytorch_device)

# 接下来，根据加载图像得到的 led_indices，从 all_kx, all_ky 中筛选出我们需要的k-vectors
# 注意：我们的 led_indices 是从1开始的，所以需要减1来作为张量索引
indices_for_slicing = torch.tensor(loaded_led_indices, dtype=torch.long) - 1

kx_estimated = -all_kx[indices_for_slicing]
ky_estimated = -all_ky[indices_for_slicing]

print(f"\nSelected {len(kx_estimated)} k-vectors corresponding to loaded images.")

# C. 创建光瞳初始猜测
pupil_radius_pixels = (NA_OBJECTIVE / (WAVELENGTH_NM * 1e-9)) * recon_pixel_size_m * output_size
print(f"Calculated initial pupil radius: {pupil_radius_pixels:.1f} pixels")
pupil_guess = create_circular_pupil((output_size, output_size), radius=int(pupil_radius_pixels))

# ==================== 角度可视化调用 ====================
# 在开始重建前，调用验证函数
visualize_kspace_and_captures(
    captures=captures,
    kx_normalized=kx_estimated,
    ky_normalized=ky_estimated,
    arrow_scale=500.0 
)
# ========================================================

# D. 创建物体初始猜测
object_guess = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)


# --- 5. 运行FPM重建 ---
# 配置自动停止参数
USE_AUTO_STOP = True # 开关
# ----------------------------------------------------
print("\nStarting FPM reconstruction on real data....")
reconstructed_object, reconstructed_pupil, metrics = solve_inverse(
    captures=captures,
    object=object_guess,
    pupil=pupil_guess,
    kx_batch=kx_estimated,
    ky_batch=ky_estimated,
    learn_pupil=True,       # 必须开启以校正像差
    learn_k_vectors=False,   # 强烈推荐开启以修正k-vector误差
    # 如果启用自动停止，设置 patience (例如 20)。如果是 False，传入 None
    patience=20 if USE_AUTO_STOP else None, 
    # 最大的 Epochs 仍然需要设置作为上限
    epochs=1000 if USE_AUTO_STOP else 500,
    min_delta = 1e-2  
    #vis_interval = 10
)
print("Reconstruction finished.")


# --- 6. 可视化结果 & 保存结果 ---
# ----------------------------------------------------

# --- NEW: Save Metrics to JSON ---
metrics_file_path = "output/metrics.json"
print(f"Saving metrics to {metrics_file_path}...")
with open(metrics_file_path, 'w') as f:
    json.dump(metrics, f, indent=4)
# ---------------------------------

# A. 可视化损失曲线
plt.figure(figsize=(10, 5))
plt.plot(metrics['loss'])
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.grid(True)
plt.savefig("output/real_data_loss_curve.png")

# B. 可视化重建的物体
final_amplitude = torch.abs(reconstructed_object)
final_phase = torch.angle(reconstructed_object)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
im1 = axes[0].imshow(final_amplitude.cpu().detach().numpy(), cmap='gray')
axes[0].set_title("Reconstructed Amplitude")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(final_phase.cpu().detach().numpy(), cmap='viridis')
axes[1].set_title("Reconstructed Phase")
fig.colorbar(im2, ax=axes[1])

plt.suptitle("Final Reconstruction", fontsize=16)
plt.savefig("output/final_reconstruction.png")

# C. 保存最终无损/纯净图
final_amp = torch.abs(reconstructed_object).cpu().detach().numpy()
final_phs = torch.angle(reconstructed_object).cpu().detach().numpy()

plt.imsave("output/final_amplitude_only.png", final_amp, cmap='gray')
plt.imsave("output/final_phase_only.png", final_phs, cmap='viridis')

# D. 可视化学习到的光瞳 
learned_pupil_amp = torch.abs(reconstructed_pupil)
learned_pupil_phase = torch.angle(reconstructed_pupil)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
im3 = axes[0].imshow(learned_pupil_amp.cpu().detach().numpy(), cmap='gray')
axes[0].set_title("Learned Pupil Amplitude")
fig.colorbar(im3, ax=axes[0])

im4 = axes[1].imshow(learned_pupil_phase.cpu().detach().numpy(), cmap='viridis')
axes[1].set_title("Learned Pupil Phase (System Aberration)")
fig.colorbar(im4, ax=axes[1])

plt.suptitle("Learned Pupil Function", fontsize=16)
plt.savefig("output/learned_pupil.png")

print("\nAll plots saved to 'output' folder.")

#print("Visualizing results...")
#plt.show() # 如果你想在运行时弹出所有窗口

