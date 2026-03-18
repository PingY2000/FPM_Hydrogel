import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

from inverse import solve_inverse, calculate_k_vectors_from_positions
from utils import get_default_device, create_circular_pupil

# ==========================================
# 1. 仿真系统参数配置
# ==========================================
WAVELENGTH_M = 525e-9          
NA_OBJECTIVE = 0.5            
MAGNIFICATION = 20.0           
CAMERA_PIXEL_SIZE_UM = 3.45    
DOWNSAMPLE_FACTOR = 1          
LED_POSITIONS_FILE = r"LedPosHis\led_positions 0210.csv" 

recon_pixel_size_m = (CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION) / DOWNSAMPLE_FACTOR
N_RECON = 256 

device = get_default_device()
torch.set_default_device(device)
print(f"[{device}] 启动 4.4.3 CG 唯象合成数据 (SSAA 抗锯齿版) + FPM 逆向重构仿真...")

# --- CG 模型全局参数 ---
BACKGROUND_INTENSITY_BF = 0.5 
BACKGROUND_INTENSITY_DF = 0.02 

# 定义多微球阵列: [(x_center, y_center, radius), ...] 坐标原点为图像中心 (0,0)
SPHERES = [
    (0, 45, 16),    # 偏右上的中等球
    (-50, -30, 12),  # 偏左下的小球
    (45, -35, 20)    # 偏右下的大球
]

# 超采样抗锯齿因子 (Supersampling Anti-Aliasing Factor)
# 在高维网格渲染以彻底消除亚像素高光跳动，最后严格下采样
AA_FACTOR = 4  

# ==========================================
# 2. 具备超采样与生物扰动的高精度 CG 响应核
# ==========================================
def generate_fpm_cg_capture(kx, ky, target_size, spheres, aa_factor):
    # 渲染分辨率域
    hr_size = target_size * aa_factor
    
    NA_ill = WAVELENGTH_M * np.sqrt(kx**2 + ky**2) /recon_pixel_size_m
    print(NA_ill)
    is_dark_field = NA_ill > NA_OBJECTIVE
    tilt_strength = min(NA_ill * 2.5, 1.0) 
    phi = np.arctan2(ky, kx)

    # 构建高分辨率物理坐标网格
    x = np.arange(hr_size) - hr_size // 2
    y = np.arange(hr_size) - hr_size // 2
    X, Y = np.meshgrid(x, y)

    bg_val = BACKGROUND_INTENSITY_DF if is_dark_field else BACKGROUND_INTENSITY_BF
    image_hr = np.ones((hr_size, hr_size), dtype=np.float64) * bg_val

    # 生成全局的细胞内质低频扰动 (模拟细胞器分布不均)
    # 使用高斯滤波平滑白噪声，获得空间连续的生物学异质性
    structural_noise = gaussian_filter(np.random.randn(hr_size, hr_size), sigma=10 * aa_factor) * 0.15

    for x_c, y_c, r in spheres:
        # 将靶标坐标与尺寸映射至高维渲染网格
        hr_x_c = x_c * aa_factor
        hr_y_c = y_c * aa_factor
        hr_r = r * aa_factor
        
        R_sq = (X - hr_x_c)**2 + (Y - hr_y_c)**2
        mask = R_sq <= hr_r**2

        if not np.any(mask):
            continue

        normalized_r = np.sqrt(R_sq[mask]) / hr_r
        Z = np.sqrt(np.maximum(hr_r**2 - R_sq[mask], 0))
        Z_safe = np.maximum(Z, 0.1)

        kernel = np.zeros_like(X[mask], dtype=np.float64)

        # 提取当前微球内部的物理扰动
        local_perturbation = structural_noise[mask]

        # --- 模式 1: 明场 / 斜照明 ---
        if not is_dark_field:
            # 1. 边界暗环与细胞质吸收扰动
            transmission = 1.0 - 0.8 * (normalized_r ** 4) + local_perturbation * 0.2
            
            # 2. 浮雕效应 (受局部折射率异质性扰动)
            gradient_dir = -((X[mask] - hr_x_c) * np.cos(phi) + (Y[mask] - hr_y_c) * np.sin(phi)) / Z_safe
            emboss_effect = 0.25 * tilt_strength * gradient_dir * (1.0 + local_perturbation)
            
            # 3. 亚像素微透镜聚光斑 (🎯 核心修改：让光斑变得柔和、宽广)
            shift_x = hr_r * 0.4 * tilt_strength * np.cos(phi)
            shift_y = hr_r * 0.4 * tilt_strength * np.sin(phi)
            
            # 将 sigma 从 0.25 放大到 0.50，使光斑覆盖半径的一半
            spot_sigma = hr_r * 0.50 
            # 将中心最高强度从 0.4 降到 0.20，模拟弱透镜的能量分散
            focal_spot = 0.30 * np.exp(-((X[mask] - hr_x_c - shift_x)**2 + (Y[mask] - hr_y_c - shift_y)**2) / (2 * spot_sigma**2))
            
            kernel = transmission + emboss_effect + focal_spot
            kernel = np.clip(kernel, 0.1, 1.5)
            image_hr[mask] = image_hr[mask] * kernel

        # --- 模式 2: 暗场 ---
        else:
            # 1. 强散射边缘环 (受局部散斑扰动)
            ring_width = 0.08 
            edge_ring = 1.0 * np.exp(-((normalized_r - 0.98)**2) / (2 * ring_width**2)) * (1.0 + local_perturbation)
            
            # 2. 高频不对称散射增强 (🎯 核心修改：使暗场的月牙亮斑也不那么刺眼)
            shift_x = hr_r * 0.95 * tilt_strength * np.cos(phi)
            shift_y = hr_r * 0.95 * tilt_strength * np.sin(phi)
            
            # 将 sigma 从 0.15 放大到 0.25
            spot_sigma = hr_r * 0.25 
            # 将峰值强度从 1.2 降到 0.8
            scattering_spot = 0.8 * np.exp(-((X[mask] - hr_x_c - shift_x)**2 + (Y[mask] - hr_y_c - shift_y)**2) / (2 * spot_sigma**2))
            
            kernel = 0.5 * edge_ring + scattering_spot
            kernel = np.clip(kernel, 0.0, 1.3)
            image_hr[mask] = image_hr[mask] + kernel

    # 统一平滑模拟离焦退化
    image_hr = gaussian_filter(image_hr, sigma=1.5 * aa_factor)
    
    # 🎯 极其关键：通过精确的 Block-Average 将高维画布下采样，彻底消除高频锯齿与跳动
    image_lr = image_hr.reshape(target_size, aa_factor, target_size, aa_factor).mean(axis=(1, 3))
    
    # 注入相机散粒噪声与读出噪声
    noise_level = 0.01 if is_dark_field else 0.02
    sensor_noise = np.random.normal(0, noise_level, image_lr.shape)
    image_final = np.clip(image_lr + sensor_noise, 0, 1)
    
    return image_final

# ==========================================
# 3. 生成 CG 仿真数据集 (输出抗锯齿序列)
# ==========================================
df = pd.read_csv(LED_POSITIONS_FILE)
loaded_led_indices = list(range(1, len(df) + 1))
led_coords_ideal, kx_ideal, ky_ideal = calculate_k_vectors_from_positions(
    filepath=LED_POSITIONS_FILE, lambda_nm=WAVELENGTH_M * 1e9,
    magnification=MAGNIFICATION, camera_pixel_size_um=CAMERA_PIXEL_SIZE_UM,
    recon_pixel_size_m=recon_pixel_size_m, loaded_led_indices=loaded_led_indices,
    device=device, center_led_index=1
)

pupil_radius_pixels = (NA_OBJECTIVE / WAVELENGTH_M) * recon_pixel_size_m * N_RECON
gt_pupil = create_circular_pupil((N_RECON, N_RECON), radius=int(pupil_radius_pixels)).to(device)

print("正在渲染多靶标 CG 风格多角度原始数据 (启用 4x SSAA)...")
simulated_captures_list = []

save_dir_captures = "output_simulation/sim_4_4_3_thick_3D/cg_captures"
os.makedirs(save_dir_captures, exist_ok=True)

kx_np = kx_ideal.cpu().numpy()
ky_np = ky_ideal.cpu().numpy()

for i in range(len(kx_np)):
    img_np = generate_fpm_cg_capture(kx_np[i], ky_np[i], N_RECON, SPHERES, AA_FACTOR)
    
    kx_val = kx_np[i]
    ky_val = ky_np[i]
    filename = f"cg_capture_{i:03d}.png"
    filepath = os.path.join(save_dir_captures, filename)
    plt.imsave(filepath, img_np, cmap='gray', vmin=0, vmax=1)
        
    simulated_captures_list.append(torch.tensor(img_np, dtype=torch.float32, device=device))

simulated_captures_3d = torch.stack(simulated_captures_list, dim=0)

# ==========================================
# 4. 逆向重建 (暴击二维 FPM 算法的物理死穴)
# ==========================================
print("\n将多靶标 CG 合成厚样品数据输入二维 FPM 算法进行逆向重建...")
center_idx = torch.argmin(kx_ideal**2 + ky_ideal**2)
obj_guess_amp = F.interpolate(simulated_captures_3d[center_idx:center_idx+1].unsqueeze(0), size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)

# 确保接收光瞳函数 (第二个返回值)
recon_object, recon_pupil, _, _, metrics, _, _, _ = solve_inverse(
    captures=simulated_captures_3d, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=led_coords_ideal, wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=True, learn_k_vectors=False, epochs=500, vis_interval=20
)

# ==========================================
# 5. 重建探针解耦提取与多靶标论证大图
# ==========================================
print("\n提取并独立保存解算出的物函数与光瞳复振幅...")
recon_save_dir = "output_simulation/sim_4_4_3_thick_3D/reconstruction_details"
os.makedirs(recon_save_dir, exist_ok=True)

# 计算振幅与相位
recon_obj_amp = torch.abs(recon_object).cpu().numpy()
recon_obj_phase = torch.angle(recon_object).cpu().numpy()
recon_pupil_amp = torch.abs(recon_pupil).cpu().numpy()
recon_pupil_phase = torch.angle(recon_pupil).cpu().numpy()

# 单独落盘
plt.imsave(os.path.join(recon_save_dir, "recon_object_amplitude.png"), recon_obj_amp, cmap='gray')
plt.imsave(os.path.join(recon_save_dir, "recon_object_phase.png"), recon_obj_phase, cmap='viridis')
plt.imsave(os.path.join(recon_save_dir, "recon_pupil_amplitude.png"), recon_pupil_amp, cmap='gray')
plt.imsave(os.path.join(recon_save_dir, "recon_pupil_phase.png"), recon_pupil_phase, cmap='viridis')

# 生成多靶标理论上的纯解析 3D 绝对相位投影 (GT)
y, x = np.ogrid[-N_RECON//2:N_RECON//2, -N_RECON//2:N_RECON//2]
ideal_projected_phase = np.zeros((N_RECON, N_RECON))
for x_c, y_c, r in SPHERES:
    r_sq = (x - x_c)**2 + (y - y_c)**2
    mask_gt = r_sq <= r**2
    # 绝对厚度相位线性积分映射
    ideal_projected_phase[mask_gt] += 4.0 * np.pi * np.sqrt(1 - r_sq[mask_gt]/(r**2))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Degradation Topography of Multi-Spheroid System in 2D FPM", fontsize=16, fontweight='bold')

axes[0].imshow(simulated_captures_3d[center_idx].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Synthetic Brightfield\n(Shows Microlens Focus & Bio-noise)", fontsize=12)

im_gt = axes[1].imshow(ideal_projected_phase, cmap='viridis')
axes[1].set_title(f"Ideal Absolute 3D Phase\n(Max Phase: {ideal_projected_phase.max():.2f} rad)", fontsize=12)
fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

im_amp = axes[2].imshow(recon_obj_amp, cmap='gray')
axes[2].set_title("Reconstructed Amplitude\n(Distorted by Defocus/Scattering)", fontsize=12)
fig.colorbar(im_amp, ax=axes[2], fraction=0.046, pad=0.04)

im_ph = axes[3].imshow(recon_obj_phase, cmap='viridis')
axes[3].set_title(f"Reconstructed Equivalent Phase\n(Severely Wrapped & Compressed)", fontsize=12)
fig.colorbar(im_ph, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig("output_simulation/sim_4_4_3_thick_3D/cg_multi_spheroid_degradation.png", dpi=200)

print(f">>> 物理验证全部落盘！独立光瞳/物体解算探针已存放至: {recon_save_dir}")
plt.show()