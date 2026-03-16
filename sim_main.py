# sim_main.py

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你现有的模块 (确保这些文件在同级目录)
from forward import forward_model
from inverse import solve_inverse
from utils import get_default_device, create_circular_pupil
from refocus import estimate_defocus_from_phase_correlation

# ==========================================
# 1. 仿真参数配置 (上帝视角的物理设定)
# ==========================================
WAVELENGTH_NM = 525           # 波长 525 nm
NA_OBJECTIVE = 0.5            # 物镜 NA
CAMERA_PIXEL_SIZE_UM = 3.45   # 相机像素大小
MAGNIFICATION = 10.0          # 放大倍率
DOWNSAMPLE_FACTOR = 4         # 下采样倍数 (仿真低分辨率)

# 物理常数计算
wavelength_m = WAVELENGTH_NM * 1e-9
# 物方实际等效像素大小
original_pixel_size_m = (CAMERA_PIXEL_SIZE_UM * 1e-6) / MAGNIFICATION 
# 高分辨率重建的像素大小
recon_pixel_size_m = original_pixel_size_m / DOWNSAMPLE_FACTOR  

# 图像尺寸
OUTPUT_SIZE = 128  # 高分辨率大小 128x128
CAPTURE_SIZE = OUTPUT_SIZE // DOWNSAMPLE_FACTOR # 低分辨率大小 32x32

# 我们要故意加入的“物理缺陷”
KNOWN_DEFOCUS_UM = 2.5  # 故意让显微镜失焦 2.5 微米

device = get_default_device()
torch.set_default_device(device)
print(f"--- FPM Simulation started on {device} ---")

# ==========================================
# 2. 生成 Ground Truth (真实的高分辨复数物体)
# ==========================================
print("1. Generating Ground Truth Object & Pupil...")
y, x = torch.meshgrid(torch.linspace(-1, 1, OUTPUT_SIZE), torch.linspace(-1, 1, OUTPUT_SIZE), indexing='ij')
r = torch.sqrt(x**2 + y**2)

# 振幅：模拟细胞轮廓 (一个环加上一些内部结构)
gt_amp = 0.4 + 0.6 * torch.exp(-10 * r**2) + 0.3 * torch.sin(15 * x) * (r < 0.6)
gt_amp = torch.clamp(gt_amp, 0.1, 1.0) # 振幅限制在 0~1

# 相位：模拟细胞厚度/折射率引起的相位延迟
gt_phase = 1.5 * torch.exp(-20 * ((x-0.2)**2 + (y+0.2)**2)) - 0.8 * torch.exp(-15 * ((x+0.3)**2 + (y-0.1)**2))
gt_object = (gt_amp * torch.exp(1j * gt_phase)).to(dtype=torch.complex64)

# 生成理想的圆形光瞳
pupil_radius_pixels = (NA_OBJECTIVE / wavelength_m) / recon_pixel_size_m * OUTPUT_SIZE
gt_pupil = create_circular_pupil((OUTPUT_SIZE, OUTPUT_SIZE), radius=int(pupil_radius_pixels)).to(device)

# ==========================================
# 3. 模拟 LED 阵列与物理成像过程 (Forward Pass)
# ==========================================
print(f"2. Simulating Forward Imaging (Adding {KNOWN_DEFOCUS_UM} um defocus into PUPIL)...")

# 【核心升级】：9x9 阵列 (81张图)，最大 NA 扩展到 0.7，引入极其关键的暗场数据！
na_grid = torch.linspace(-0.7, 0.7, 9, device=device)
NA_Y, NA_X = torch.meshgrid(na_grid, na_grid, indexing='ij')
NA_X, NA_Y = NA_X.flatten(), NA_Y.flatten()

kx_gt = NA_X / wavelength_m * recon_pixel_size_m
ky_gt = NA_Y / wavelength_m * recon_pixel_size_m

# 生成频率网格用于计算离焦相位差
fy = torch.arange(OUTPUT_SIZE, dtype=torch.float32, device=device) - OUTPUT_SIZE // 2
fx = torch.arange(OUTPUT_SIZE, dtype=torch.float32, device=device) - OUTPUT_SIZE // 2
fy = fy / (OUTPUT_SIZE * recon_pixel_size_m)
fx = fx / (OUTPUT_SIZE * recon_pixel_size_m)
FY, FX = torch.meshgrid(fy, fx, indexing='ij')

# 菲涅尔离焦相位: exp(i * pi * lambda * z * (fx^2 + fy^2))
defocus_phase = torch.pi * wavelength_m * (KNOWN_DEFOCUS_UM * 1e-6) * (FX**2 + FY**2)
defocused_pupil = gt_pupil * torch.exp(1j * defocus_phase)

# 生成图像时：使用【完美的物体】+【带有离焦像差的光瞳】
clean_captures = forward_model(gt_object, defocused_pupil, kx_gt, ky_gt, DOWNSAMPLE_FACTOR)

# 加入泊松噪声和高斯读出噪声
peak_photons = 8000.0
noisy_captures = torch.poisson(clean_captures / clean_captures.max() * peak_photons) / peak_photons
noisy_captures += torch.randn_like(noisy_captures) * 0.01 
captures = torch.clamp(noisy_captures, 0.0) # [81, 32, 32]

# ==========================================
# 4. 测试 Autofocus 算法
# ==========================================
print("\n3. Testing Phase Correlation Autofocus...")
# 9x9=81张图，正中心的那张索引是 40，相邻的一张是 41
idx_center = 40
idx_tilted = 41

est_z_m = estimate_defocus_from_phase_correlation(
    img_center=captures[idx_center],
    img_tilted=captures[idx_tilted],
    na_x=NA_X[idx_tilted].item(),
    na_y=NA_Y[idx_tilted].item(),
    pixel_size_obj_m=original_pixel_size_m
)
print(f"   => Ground Truth Defocus: {KNOWN_DEFOCUS_UM:.3f} um")
print(f"   => Estimated Defocus:    {est_z_m * 1e6:.3f} um")
print(f"   => Error:                {abs(KNOWN_DEFOCUS_UM - est_z_m*1e6):.3f} um")

# ==========================================
# 5. 将算出的离焦注入到“初始光瞳”中 (核心杀招)
# ==========================================
print("\n4. Injecting Estimated Defocus into Initial Pupil...")
# 利用估算出的 z 构建初始抛物线相位，赋予 pupil_guess
est_defocus_phase = torch.pi * wavelength_m * est_z_m * (FX**2 + FY**2)
pupil_guess = gt_pupil * torch.exp(1j * est_defocus_phase)

# ==========================================
# 6. 运行 FPM 逆向重建
# ==========================================
print("\n5. Running FPM Reconstruction with Pre-corrected Pupil...")
# 初始物体猜测
obj_guess_amp = F.interpolate(captures[idx_center:idx_center+1, None, :, :], size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bicubic')[0, 0]
obj_guess = obj_guess_amp.to(torch.complex64)

reconstructed_object, reconstructed_pupil, _, _, metrics, _, _, _ = solve_inverse(
    captures=captures,
    object=obj_guess,
    pupil=pupil_guess,      # <--- 喂入带有我们算出的离焦相位的初始光瞳！
    led_physics_coords=None,
    wavelength=wavelength_m,
    recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_gt,
    ky_batch=ky_gt,
    learn_pupil=True,       # <--- 允许算法微调残余的微小误差
    learn_k_vectors=False, 
    epochs=150,
    vis_interval=0
)

# ==========================================
# 7. 计算最终误差并画图 (消除全局相位漂移)
# ==========================================
def calc_mse(recon, gt):
    phase_diff = torch.angle(recon) - torch.angle(gt)
    recon_aligned = recon * torch.exp(-1j * torch.mean(phase_diff))
    return torch.mean(torch.abs(recon_aligned - gt)**2).item()

mse_final = calc_mse(reconstructed_object, gt_object)
print(f"   => Final MSE: {mse_final:.5f}")

print("\n6. Plotting Results...")
# === 消除相位的全局常数漂移，对齐背景 ===
recon_phase = torch.angle(reconstructed_object)
recon_phase = recon_phase - torch.mean(recon_phase)

gt_phase = torch.angle(gt_object)
gt_phase = gt_phase - torch.mean(gt_phase)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(torch.abs(gt_object).cpu().numpy(), cmap='gray'); axes[0, 0].set_title("1. Ground Truth (Amplitude)")
axes[0, 1].imshow(gt_phase.cpu().numpy(), cmap='viridis'); axes[0, 1].set_title("2. Ground Truth (Phase)")

axes[1, 0].imshow(torch.abs(reconstructed_object).cpu().numpy(), cmap='gray'); axes[1, 0].set_title(f"3. FPM with Autofocus Prior\nMSE: {mse_final:.4f}")
axes[1, 1].imshow(recon_phase.cpu().numpy(), cmap='viridis'); axes[1, 1].set_title("4. Reconstructed Phase")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/simulation_result.png", dpi=150)
print("Simulation completed! Check 'output/simulation_result.png' for visual proof.")