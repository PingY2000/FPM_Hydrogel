import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from inverse import solve_inverse, calculate_k_vectors_from_positions
from utils import get_default_device, create_circular_pupil

# ==========================================
# 1. 仿真系统参数配置
# ==========================================
WAVELENGTH_M = 525e-9          
NA_OBJECTIVE = 0.1            
MAGNIFICATION = 4.0           
CAMERA_PIXEL_SIZE_UM = 3.45    
DOWNSAMPLE_FACTOR = 1  # 解析投影模型不依赖数值切片采样，可以直接使用 1        
LED_POSITIONS_FILE = r"LedPosHis\led_positions 0210.csv" 

recon_pixel_size_m = (CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION) / DOWNSAMPLE_FACTOR
N_RECON = 256 

device = get_default_device()
torch.set_default_device(device)
print(f"[{device}] 启动 4.4.3 CG+物理解析射线投影仿真 (战略性破局)...")

# ==========================================
# 2 & 3. 唯象物理建模：基于解析射线追踪的前向投影近似 (Analytical Ray-Tracing)
# ==========================================
print("执行解析几何射线追踪 (纯数学推演，确保多角度视差绝对正确)...")

# 还原生物特征参数
RADIUS_UM = 15.0
R_m = RADIUS_UM * 1e-6
delta_n = 0.035  # 真实的活细胞折射率失配
k0 = 2 * np.pi / WAVELENGTH_M
Z_c = 0.0 # 微球球心坐标位于焦平面

# 严格的零点居中绝对物理坐标系 (单位：米)
y_idx = torch.arange(N_RECON, device=device, dtype=torch.float32) - N_RECON / 2.0
x_idx = torch.arange(N_RECON, device=device, dtype=torch.float32) - N_RECON / 2.0
Y_grid, X_grid = torch.meshgrid(y_idx * recon_pixel_size_m, x_idx * recon_pixel_size_m, indexing='ij')

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

simulated_captures_3d = []

# ================= 核心：解析几何射线投影 =================
with torch.no_grad():
    for i in range(len(kx_ideal)):
        kx, ky = kx_ideal[i], ky_ideal[i]
        
        # 1. 计算当前照明方向矢量
        ux = kx * WAVELENGTH_M*1e6
        uy = ky * WAVELENGTH_M*1e6
        uz = torch.sqrt(torch.clamp(1.0 - ux**2 - uy**2, 0.0))
        
        # 2. 计算射线到球心的垂直距离平方 (叉乘几何法，彻底消除网格混叠波浪纹)
        cp_x = Y_grid * uz - (-Z_c) * uy
        cp_y = (-Z_c) * ux - X_grid * uz
        cp_z = X_grid * uy - Y_grid * ux
        d_sq = cp_x**2 + cp_y**2 + cp_z**2
        
        # 3. 弦长计算 (物理厚度 L)
        chord_length_m = 2.0 * torch.sqrt(torch.clamp(R_m**2 - d_sq, min=0.0))
        
        # 4. 相位累积与唯象吸收 (模拟 CG 代码里的边界暗环效应)
        phase_delay = k0 * delta_n * chord_length_m
        
        # 边缘吸收：L 越短(越靠近边缘)，吸收越多，模拟唯象的阴影效果
        # normalized_l 是 L 在 R_m 方向的投影
        normalized_l = chord_length_m / (2.0 * R_m)
        amp = 0.98 * (1.0 - 0.2 * torch.pow(torch.clamp(1.0 - normalized_l, 0, 1), 4))
        
        # 5. 生成出射波前 (倾斜照明载波 * 物质相互作用项)
        # 此时 exit_wave 是极致光滑的牛顿环相位分布！没有任何数字锯齿。
        exit_wave = amp * torch.exp(1j * (2 * np.pi * (kx * X_grid + ky * Y_grid) + phase_delay))
        
        # 6. 过光瞳 (低通滤波)，形成像面光场
        # 🎯 这是产生聚光点和月牙的关键：傅里叶滤波将纯相位转换为强度起伏！
        field_fourier = fft.fftshift(fft.fft2(exit_wave))
        filtered_fourier = gt_pupil * field_fourier
        image_field = fft.ifft2(fft.ifftshift(filtered_fourier))
        
        # 7. 相机记录强度
        intensity = torch.abs(image_field)**2
        
        simulated_captures_3d.append(intensity)
        
simulated_captures_3d = torch.stack(simulated_captures_3d, dim=0)

# 直接利用球冠解析式计算理想状况下 3D 球体的积分绝对相位 (作为重构的参照)
ideal_projected_phase = (k0 * delta_n * 2.0 * torch.sqrt(torch.clamp(R_m**2 - (X_grid**2 + Y_grid**2), min=0.0))).cpu().numpy()

# ==========================================
# 3.5 导出高保真多角度物理原图
# ==========================================
save_dir_captures = "output_simulation/sim_4_4_3_thick_3D/simulated_captures"
os.makedirs(save_dir_captures, exist_ok=True)
captures_3d_np = simulated_captures_3d.cpu().numpy()

for i in range(len(captures_3d_np)):
    img = captures_3d_np[i]
    # 局部极值归一化，把暗场微弱的月牙偏折硬拉出来！
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    kx_val, ky_val = kx_ideal[i].item(), ky_ideal[i].item()
    plt.imsave(os.path.join(save_dir_captures, f"capture_{i:03d}_kx{kx_val:+.3f}_ky{ky_val:+.3f}.png"), img_norm, cmap='gray')
    
print(f">>> 解析射线模型演算完毕！物理原图已导出至: {save_dir_captures}")

# ==========================================
# 4. 采用二维 FPM 模型进行逆向重建 (暴击失效边界)
# ==========================================
print("\n启动传统 2D FPM 重建 (强行将 3D 微球特征压平)...")
center_idx = torch.argmin(kx_ideal**2 + ky_ideal**2)
obj_guess_amp = F.interpolate(simulated_captures_3d[center_idx:center_idx+1].unsqueeze(0), size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)

# 此处仍开启刚体校准，保证参数最优，重点看 2D 模型的物理崩塌
recon_object, _, _, _, metrics, _, _, _ = solve_inverse(
    captures=simulated_captures_3d, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=led_coords_ideal, wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=False, learn_k_vectors=True, epochs=300, vis_interval=0
)

# ==========================================
# 5. 分析“等效投影相位”效应并出图
# ==========================================
print("\n>>> 计算完成！正在分析单体厚微球的相位压缩退化...")
recon_amp = torch.abs(recon_object).cpu().numpy()
recon_phase = torch.angle(recon_object).cpu().numpy()

os.makedirs("output_simulation/sim_4_4_3_thick_3D", exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("2D FPM Degradation on an Isolated 3D SKOV-3 Spheroid", fontsize=16, fontweight='bold')

# 1. 低分辨率原图 (观察厚细胞的三维离焦边缘)
axes[0].imshow(simulated_captures_3d[center_idx].cpu().numpy(), cmap='gray')
axes[0].set_title("Brightfield Capture\n(Smooth focus projection)", fontsize=12)

# 2. 理想 2D 积分绝对相位
im_gt = axes[1].imshow(ideal_projected_phase, cmap='viridis')
axes[1].set_title(f"Ideal 3D Integrated Phase\n(Max Phase: {ideal_projected_phase.max():.2f} rad)", fontsize=12)
fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

# 3. 2D FPM 重建振幅 (暴露串扰伪影)
im_amp = axes[2].imshow(recon_amp, cmap='gray')
axes[2].set_title("Reconstructed Equivalent Amp\n(Edge Diffraction Artifacts)", fontsize=12)
fig.colorbar(im_amp, ax=axes[2], fraction=0.046, pad=0.04)

# 4. 2D FPM 重建等效相位 (暴露出严重的相位包裹与非线性压缩)
im_ph = axes[3].imshow(recon_phase, cmap='viridis')
axes[3].set_title(f"Equivalent Projected Phase\n(Wrapped & Compressed, Max: {recon_phase.max():.2f} rad)", fontsize=12)
fig.colorbar(im_ph, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
save_path = "output_simulation/sim_4_4_3_thick_3D/isolated_spheroid_degradation.png"
plt.savefig(save_path, dpi=200)

# 单独导出论文图
plt.imsave("output_simulation/sim_4_4_3_thick_3D/spheroid_ideal_phase.png", ideal_projected_phase, cmap='viridis')
plt.imsave("output_simulation/sim_4_4_3_thick_3D/spheroid_recon_phase.png", recon_phase, cmap='viridis')

print(f"\n>>> 仿真与重构全线打通！结果已保存至: {save_path}")
plt.show()