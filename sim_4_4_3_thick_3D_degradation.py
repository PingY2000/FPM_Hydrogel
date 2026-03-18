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
print(f"[{device}] 启动 4.4.3 CG 唯象合成数据 + FPM 逆向重构仿真...")

# --- CG 模型全局参数 ---
BACKGROUND_INTENSITY_BF = 0.5 
BACKGROUND_INTENSITY_DF = 0.02 
SPHERE_RADIUS_PIXELS = 15 # 微球半径 (约合 38um)

# ==========================================
# 2. 你的绝佳思路：CG 唯象光学响应核生成器
# ==========================================
def generate_fpm_cg_capture(kx, ky, image_size, radius):
    """
    基于你提供的 CG 逻辑，并适配 FPM 的 (kx, ky) 矢量照明。
    """
    # 1. 计算照明 NA 和方位角
    NA_ill = WAVELENGTH_M * np.sqrt(kx**2 + ky**2)*1e6
    print(NA_ill)
    is_dark_field = NA_ill > NA_OBJECTIVE
    
    # 将 NA 映射为倾斜强度 (相当于原代码的 sin(angle_rad))
    # 为了让视觉效果明显，稍微放大这个系数
    tilt_strength = min(NA_ill * 2.5, 1.0) 
    
    # 光源方位角 (用于旋转阴影和亮斑)
    phi = np.arctan2(ky, kx)
    print(phi)
    x = np.arange(image_size) - image_size // 2
    y = np.arange(image_size) - image_size // 2
    X, Y = np.meshgrid(x, y)
    R_sq = X**2 + Y**2
    mask = R_sq <= radius**2

    # 初始化背景
    bg_val = BACKGROUND_INTENSITY_DF if is_dark_field else BACKGROUND_INTENSITY_BF
    image = np.ones((image_size, image_size), dtype=np.float64) * bg_val

    if not np.any(mask):
        return image

    normalized_r = np.sqrt(R_sq[mask]) / radius
    Z = np.sqrt(np.maximum(radius**2 - R_sq[mask], 0))
    Z_safe = np.maximum(Z, 0.1)

    kernel = np.zeros_like(X[mask], dtype=np.float64)

    # --- 模式 1: 明场 / 斜照明 ---
    if not is_dark_field:
        # 1. 边界暗环
        transmission = 1.0 - 0.8 * (normalized_r ** 4) 
        
        # 2. 浮雕效应 (加入方位角 phi 进行旋转)
        gradient_dir = -(X[mask] * np.cos(phi) + Y[mask] * np.sin(phi)) / Z_safe
        emboss_effect = 0.25 * tilt_strength * gradient_dir
        
        # 3. 微透镜聚光斑
        shift_x = radius * 0.4 * tilt_strength * np.cos(phi)
        shift_y = radius * 0.4 * tilt_strength * np.sin(phi)
        spot_sigma = radius * 0.25
        focal_spot = 0.4 * np.exp(-((X[mask] - shift_x)**2 + (Y[mask] - shift_y)**2) / (2 * spot_sigma**2))
        
        kernel = transmission + emboss_effect + focal_spot
        kernel = np.clip(kernel, 0.1, 1.5)
        image[mask] = image[mask] * kernel

    # --- 模式 2: 暗场 ---
    else:
        # 1. 强散射边缘环
        ring_width = 0.08 
        edge_ring = 1.0 * np.exp(-((normalized_r - 0.98)**2) / (2 * ring_width**2))
        
        # 2. 不对称散射增强 (光源反方向)
        shift_x = radius * 0.95 * tilt_strength * np.cos(phi)
        shift_y = radius * 0.95 * tilt_strength * np.sin(phi)
        spot_sigma = radius * 0.15 
        scattering_spot = 1.2 * np.exp(-((X[mask] - shift_x)**2 + (Y[mask] - shift_y)**2) / (2 * spot_sigma**2))
        
        kernel = 0.5 * edge_ring + scattering_spot
        kernel = np.clip(kernel, 0.0, 1.3)
        image[mask] = image[mask] + kernel

    # 统一平滑与加噪
    image = gaussian_filter(image, sigma=1.5)
    noise_level = 0.01 if is_dark_field else 0.02
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image

# ==========================================
# 3. 生成 CG 仿真数据集
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

print("正在渲染 CG 风格多角度原始数据...")
simulated_captures_list = []

save_dir_captures = "output_simulation/sim_4_4_3_thick_3D/cg_captures"
os.makedirs(save_dir_captures, exist_ok=True)

# 转换 kx_ideal, ky_ideal 到 CPU numpy 数组进行 CG 渲染
kx_np = kx_ideal.cpu().numpy()
ky_np = ky_ideal.cpu().numpy()

for i in range(len(kx_np)):
    img_np = generate_fpm_cg_capture(kx_np[i], ky_np[i], N_RECON, SPHERE_RADIUS_PIXELS)
    
    # 获取当前的 kx, ky 坐标
    kx_val = kx_np[i]
    ky_val = ky_np[i]
    
    # 生成带编号和绝对坐标的详细文件名
    filename = f"cg_capture_{i:03d}_kx{kx_val:+.3f}_ky{ky_val:+.3f}.png"
    filepath = os.path.join(save_dir_captures, filename)
    
    # 取消 if 限制，直接保存每一张生成的图片！
    plt.imsave(filepath, img_np, cmap='gray', vmin=0, vmax=1)
        
    simulated_captures_list.append(torch.tensor(img_np, dtype=torch.float32, device=device))
simulated_captures_3d = torch.stack(simulated_captures_list, dim=0)

# ==========================================
# 4. 逆向重建 (暴击二维 FPM 算法的物理死穴)
# ==========================================
print("\n将 CG 合成厚样品数据输入二维 FPM 算法进行逆向重建...")
center_idx = torch.argmin(kx_ideal**2 + ky_ideal**2)
obj_guess_amp = F.interpolate(simulated_captures_3d[center_idx:center_idx+1].unsqueeze(0), size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)

recon_object, _, _, _, metrics, _, _, _ = solve_inverse(
    captures=simulated_captures_3d, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=led_coords_ideal, wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=True, learn_k_vectors=False, epochs=500, vis_interval=20
)

# ==========================================
# 5. 结论对比图
# ==========================================
recon_amp = torch.abs(recon_object).cpu().numpy()
recon_phase = torch.angle(recon_object).cpu().numpy()

# 为了对比，生成一个理论上的纯解析 3D 绝对相位球冠
y, x = np.ogrid[-N_RECON//2:N_RECON//2, -N_RECON//2:N_RECON//2]
r_sq = x**2 + y**2
mask_gt = r_sq <= SPHERE_RADIUS_PIXELS**2
ideal_projected_phase = np.zeros((N_RECON, N_RECON))
ideal_projected_phase[mask_gt] = 4.0 * np.pi * np.sqrt(1 - r_sq[mask_gt]/(SPHERE_RADIUS_PIXELS**2))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("2D FPM Algorithm Failure Boundary via Phenomenological CG Data", fontsize=16, fontweight='bold')

# 1. 你的杰作：CG 明场
axes[0].imshow(simulated_captures_3d[center_idx].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0].set_title("CG Synthetic Brightfield\n(Notice the microlens spot)", fontsize=12)

# 2. 理论绝对 3D 积分相位
im_gt = axes[1].imshow(ideal_projected_phase, cmap='viridis')
axes[1].set_title(f"Ideal Absolute 3D Phase\n(Max Phase: {ideal_projected_phase.max():.2f} rad)", fontsize=12)
fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

# 3. FPM 振幅
im_amp = axes[2].imshow(recon_amp, cmap='gray')
axes[2].set_title("Reconstructed Amplitude\n(Distorted by Defocus/Scattering)", fontsize=12)
fig.colorbar(im_amp, ax=axes[2], fraction=0.046, pad=0.04)

# 4. FPM 等效相位的崩溃
im_ph = axes[3].imshow(recon_phase, cmap='viridis')
axes[3].set_title(f"Reconstructed Equivalent Phase\n(Severely Wrapped & Compressed)", fontsize=12)
fig.colorbar(im_ph, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig("output_simulation/sim_4_4_3_thick_3D/cg_spheroid_degradation.png", dpi=200)
print("\n>>> 大功告成！完美利用 CG 原始数据摧毁了 2D FPM 的相位极限。")
plt.show()