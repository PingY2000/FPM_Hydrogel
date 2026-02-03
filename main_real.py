# main_real.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import ptych


# 从我们的 ptych 包和新的 utils 文件导入函数
from ptych import solve_inverse
from utils import (
    get_default_device,
    load_real_captures,
    calculate_k_vectors_from_positions,
    create_circular_pupil,
    visualize_kspace_and_captures
)


# --- 1. 用户配置 (User Configuration) ---
# 请根据你的实际实验装置和数据填写此部分！
# ----------------------------------------------------

# A. 系统物理参数
NA_OBJECTIVE = 0.5          # 物镜 NA
WAVELENGTH_NM = 532         # LED 波长 (nm)
MAGNIFICATION = 10.0        # 物镜放大倍率 (e.g., 10x)
CAMERA_PIXEL_SIZE_UM = 3.45 # 相机像素尺寸 (um)

# B. 数据和重建参数
CAPTURES_PATH = "D:\FPM_Dataset\OnTest" # 你存放真实图像的文件夹
LED_POSITIONS_FILE = "led_positions.csv" # 你的LED位置文件
CENTER_LED_INDEX = 1        # 对应中心照明的LED的索引号 (从1开始)
DOWNSAMPLE_FACTOR = 1       # 你的 captures 是否被下采样了？如果是256x256，可能不需要再下采样，设为1
                            # 如果你的 forward_model 内部有下采样，要匹配起来

# C. 重建超参数
EPOCHS = 100
LEARNING_RATE = 0.1

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
    arrow_scale=500.0 # <-- 这是一个可调参数，如果箭头太长或太短，请修改它
)
# ========================================================

# D. 创建物体初始猜测
object_guess = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)


# --- 5. 运行FPM重建 ---
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
    # 注意: 你可能需要修改 solve_inverse 函数来接受 epochs 和 lr 作为参数
    # 如果 solve_inverse 内部写死了 epochs=100, lr=0.1, 那么这里的参数无效
)
print("Reconstruction finished.")


# --- 6. 可视化结果 ---
# ----------------------------------------------------
print("Visualizing results...")

# A. 可视化损失曲线
plt.figure(figsize=(10, 5))
plt.plot(metrics['loss'])
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.grid(True)
plt.savefig("tmp/real_data_loss_curve.png")

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
plt.savefig("tmp/final_reconstruction.png")

# C. 可视化学习到的光瞳 (非常重要!)
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
plt.savefig("tmp/learned_pupil.png")

print("\nAll plots saved to 'tmp/' folder.")
# plt.show() # 如果你想在运行时弹出所有窗口