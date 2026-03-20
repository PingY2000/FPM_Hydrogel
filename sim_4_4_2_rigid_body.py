# sim_4_4_2_rigid_body.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from forward import forward_model
from inverse import solve_inverse, calculate_k_vectors_from_positions, compute_k_from_rigid_body
from utils import get_default_device, create_circular_pupil

# ==========================================
# 1. 仿真系统参数配置
# ==========================================
WAVELENGTH_M = 525e-9          
NA_OBJECTIVE = 0.1            
MAGNIFICATION = 4.0           
CAMERA_PIXEL_SIZE_UM = 3.45    
DOWNSAMPLE_FACTOR = 1          
LED_POSITIONS_FILE = "LedPosHis\led_positions 0210.csv"

original_pixel_size_m = CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION
recon_pixel_size_m = original_pixel_size_m / DOWNSAMPLE_FACTOR
N_RECON = 512 # 为了快速验证收敛曲线，此处使用 256 尺寸即可

device = get_default_device()
torch.set_default_device(device)
print(f"[{device}] 初始化 4.4.2 刚体误差自愈收敛仿真...")

'''
# ==========================================
# 2. 生成基准物体 (简化版 USAF，保证运行速度)
# ==========================================
def generate_ground_truth(size):
    Y, X = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    amp = np.ones((size, size), dtype=np.float32) * 0.2
    phase = np.zeros((size, size), dtype=np.float32)

    def add_element(x, y, w):
        for i in range(3):
            # 水平
            y0, y1 = y + i * 2 * w, y + i * 2 * w + w
            mask = (X >= x) & (X < x + 5 * w) & (Y >= y0) & (Y < y1)
            amp[mask] = 1.0; phase[mask] = 1.0
            # 垂直
            x0, x1 = x + 6 * w + i * 2 * w, x + 6 * w + i * 2 * w + w
            mask = (X >= x0) & (X < x1) & (Y >= y) & (Y < y + 5 * w)
            amp[mask] = 1.0; phase[mask] = 1.0

    widths = [8, 5, 3, 2]
    cx, cy = 30, 30
    for w in widths:
        add_element(cx, cy, w)
        cx += 11 * w + 20
        cy += 5 * w + 20

    gt_complex = amp * np.exp(1j * phase)
    return torch.tensor(gt_complex, dtype=torch.complex64, device=device)

gt_object = generate_ground_truth(N_RECON)
'''

# ==========================================
# 2. 生成基准物体 (512x512 金字塔对称排版 USAF 1951)
# ==========================================
def generate_ground_truth(size):

    Y, X = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    """
    生成高度还原的 USAF 1951 3-bar 线对测试卡。
    采用“金字塔”式居中对称排版，极具学术严谨性，专为 512x512 优化。
    """
    amp = np.ones((size, size), dtype=np.float32) * 0.2
    phase = np.zeros((size, size), dtype=np.float32)

    def smooth_rect(x0, x1, y0, y1, eps=0.5):
        sx = 1 / (1 + np.exp(-(X - x0)/eps)) * (1 / (1 + np.exp(-(x1 - X)/eps)))
        sy = 1 / (1 + np.exp(-(Y - y0)/eps)) * (1 / (1 + np.exp(-(y1 - Y)/eps)))
        return sx * sy

    def add_element(x, y, w):
        """
        支持浮点尺寸的测试单元绘制（亚像素几何）
        左：水平条纹
        右：垂直条纹
        """
        # ===== 左侧：水平条纹 =====
        for i in range(3):
            y0 = y + i * 2 * w
            y1 = y0 + w
            x0 = x
            x1 = x + 5 * w

            mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
            amp[mask] = 1.0
            phase[mask] = 1.0

        # ===== 右侧：垂直条纹 =====
        x_vert = x + 6 * w  # 5w + w 间隔

        for i in range(3):
            x0 = x_vert + i * 2 * w
            x1 = x0 + w
            y0 = y
            y1 = y + 5 * w

            mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
            amp[mask] = 1.0
            phase[mask] = 1.0

    # 手动定义“金字塔”每一行的线宽组 (从大到小排列)
    cols = [
    [13,14,15],
    [1.0,1.2,1.5,1.8,2.2,2.6,3.0,3.6,4.2,5.0,6.0,7.0],
    [8,9,10,11,12]
]
    
    gap_x = 35  # 列间距
    alpha = 2  # 间距系数（可调）

    # ===== 计算总宽度 =====
    total_w = sum(11 * max(col) for col in cols) + (len(cols) - 1) * gap_x
    start_x = (size - total_w) // 2

    curr_x = start_x

    for col in cols:
        # ===== 先计算列总高度（动态gap）=====
        col_h = 0
        for i, w in enumerate(col):
            col_h += 5 * w
            if i < len(col) - 1:
                col_h += int(alpha * w)

        curr_y = (size - col_h) // 2


        # ===== 绘制 =====
        for i, w in enumerate(col):
            add_element(curr_x, curr_y, w)

            if i < len(col) - 1:
                w_next = col[i + 1]
                gap_y_dynamic = int(alpha * (w + w_next) / 2)
                curr_y += 5 * w + gap_y_dynamic

        curr_x += 11 * max(col) + gap_x

    gt_complex = amp * np.exp(1j * phase)
    return torch.tensor(gt_complex, dtype=torch.complex64, device=device)

gt_object = generate_ground_truth(N_RECON)


# ==========================================
# 3. 注入已知的物理刚体误差 (Ground Truth Errors)
# ==========================================
print("读取理想 LED 物理坐标...")
df = pd.read_csv(LED_POSITIONS_FILE)
loaded_led_indices = list(range(1, len(df) + 1))

# 获取理想的物理坐标 [X, Y, Z]
led_coords_ideal, kx_ideal, ky_ideal = calculate_k_vectors_from_positions(
    filepath=LED_POSITIONS_FILE, lambda_nm=WAVELENGTH_M * 1e9,
    magnification=MAGNIFICATION, camera_pixel_size_um=CAMERA_PIXEL_SIZE_UM,
    recon_pixel_size_m=recon_pixel_size_m, loaded_led_indices=loaded_led_indices,
    device=device, center_led_index=1
)

# 🎯 设定极其严酷的装配误差 (单位：米, 弧度)
# X 偏移: +1.5 mm, Y 偏移: -2.0 mm, Z 偏移: 0 mm, 旋转: +2.0 度
TRUE_DX = 0.3e-3         # 偏移 0.3 mm (300微米)
TRUE_DY = -0.8e-3        # 偏移 -0.4 mm (-400微米)
TRUE_DZ = 0.0
TRUE_THETA = np.radians(0.4) # 旋转 0.8 度

true_rigid_params = torch.tensor([TRUE_DX, TRUE_DY, TRUE_DZ, TRUE_THETA], device=device)

# 计算带有真实误差的 K 向量
kx_corrupted, ky_corrupted = compute_k_from_rigid_body(
    led_coords_ideal, true_rigid_params, WAVELENGTH_M, recon_pixel_size_m
)
print(f"注入误差: dx={TRUE_DX*1e3}mm, dy={TRUE_DY*1e3}mm, theta={np.degrees(TRUE_THETA)}度")

# ==========================================
# 4. 生成带有误差的“真实”低分辨率图
# ==========================================
pupil_radius_pixels = (NA_OBJECTIVE / WAVELENGTH_M) * recon_pixel_size_m * N_RECON
gt_pupil = create_circular_pupil((N_RECON, N_RECON), radius=int(pupil_radius_pixels)).to(device)

'''
with torch.no_grad():
    simulated_captures = forward_model(
        gt_object, gt_pupil, kx_corrupted, ky_corrupted, downsample_factor=DOWNSAMPLE_FACTOR
    )
    simulated_captures = torch.clamp(simulated_captures, min=0)
'''    

with torch.no_grad():
    simulated_captures = forward_model(
        gt_object, gt_pupil, kx_corrupted, ky_corrupted, downsample_factor=DOWNSAMPLE_FACTOR
    )
    # 添加极其微弱的本底噪声 (35dB) 以避免除零错误
    for i in range(len(kx_corrupted)):
        local_max = simulated_captures[i].max()
        noise_level = local_max * (10 ** (-35 / 20))
        simulated_captures[i] += torch.randn_like(simulated_captures[i]) * noise_level
        
    simulated_captures = torch.clamp(simulated_captures, min=0)
    simulated_captures += torch.randn_like(simulated_captures) * noise_level
    simulated_captures = torch.clamp(simulated_captures, min=0)

# ==========================================
# 5. 闭环验证：联合重建 (隐瞒误差，让算法自己找)
# ==========================================
print("\n启动联合重建，尝试盲解耦刚体误差...")
center_idx = torch.argmin(kx_ideal**2 + ky_ideal**2)
obj_guess_amp = F.interpolate(simulated_captures[center_idx:center_idx+1].unsqueeze(0), size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)

# 🛑 注意：此处喂给算法的是 led_coords_ideal (包含 0 误差的理想坐标) 
# 并开启 learn_k_vectors = True (激活 utils/inverse.py 中的刚体优化)
recon_object, _, _, _, metrics, _, _, _ = solve_inverse(
    captures=simulated_captures,
    object=obj_guess,
    pupil=gt_pupil, 
    led_physics_coords=led_coords_ideal, # 传入理想坐标
    wavelength=WAVELENGTH_M,
    recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal,                   # 传入理想 K 向量
    ky_batch=ky_ideal,
    learn_pupil=False,        
    learn_k_vectors=True,                # 开启刚体自校准
    epochs=400,                          # 需要较多迭代次数以保证刚体参数收敛 
    vis_interval=0
)
# ==========================================
# 6. 核心学术图表绘制：参数收敛曲线 (中文论文规范版)
# ==========================================
os.makedirs("output_simulation", exist_ok=True)

# 确保支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 提取优化过程中的参数记录
epochs_array = np.arange(len(metrics['dx']))
dx_history = np.array(metrics['dx']) * 1e3 # 转为 mm
dy_history = np.array(metrics['dy']) * 1e3 # 转为 mm
theta_history = np.degrees(np.array(metrics['theta'])) # 转为度

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 子图 1: DX 逼近曲线 ---
axes[0].plot(epochs_array, dx_history, label='算法反演值 $d_x$', color='blue', linewidth=2)
# 注意这里的标签数值要和你的 TRUE_DX 设定一致 (0.3mm)
axes[0].axhline(y=TRUE_DX*1e3, color='red', linestyle='--', label='预设误差值 (0.3 mm)') 
axes[0].set_xlabel("迭代次数", fontsize=12)
axes[0].set_ylabel("x轴偏移量  mm", fontsize=12)
axes[0].legend(fontsize=12, loc='upper right')
axes[0].grid(True, linestyle=':')
# 在左上角添加 (a)
axes[0].text(0.05, 1.08, "(a)", transform=axes[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# --- 子图 2: DY 逼近曲线 ---
axes[1].plot(epochs_array, dy_history, label='算法反演值 $d_y$', color='green', linewidth=2)
# 注意这里的标签数值要和你的 TRUE_DY 设定一致 (-0.8mm)
axes[1].axhline(y=TRUE_DY*1e3, color='red', linestyle='--', label='预设误差值 (-0.8 mm)')
axes[1].set_xlabel("迭代次数", fontsize=12)
axes[1].set_ylabel("y轴偏移量  mm", fontsize=12)
axes[1].legend(fontsize=12, loc='upper right')
axes[1].grid(True, linestyle=':')
# 在左上角添加 (b)
axes[1].text(0.05, 1.08, "(b)", transform=axes[1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# --- 子图 3: Theta 逼近曲线 ---
axes[2].plot(epochs_array, theta_history, label='算法反演值 $\\theta$', color='purple', linewidth=2)
# 注意这里的标签数值要和你的 TRUE_THETA 设定一致 (0.4度)
axes[2].axhline(y=np.degrees(TRUE_THETA), color='red', linestyle='--', label='预设误差值 (0.4$^\circ$)')
axes[2].set_xlabel("迭代次数", fontsize=12)
axes[2].set_ylabel("旋转角度  $^\circ$", fontsize=12)
axes[2].legend(fontsize=12, loc='upper right')
axes[2].grid(True, linestyle=':')
# 在左上角添加 (c)
axes[2].text(0.05, 1.08, "(c)", transform=axes[2].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

plt.tight_layout()
save_path_curves = "output_simulation/sim_4_4_2_rigid_body_curves.png"
plt.savefig(save_path_curves, dpi=150, bbox_inches='tight') # dpi 提高到300，增加 bbox_inches 防止标签被裁
print(f"刚体校准收敛曲线已保存至: {save_path_curves}")
plt.show()

# 打印最终误差
final_dx = dx_history[-1]
final_dy = dy_history[-1]
final_theta = theta_history[-1]

print("\n>>> 【参数反演结果评估】 <<<")
print(f"真实 dx: {TRUE_DX*1e3:.3f} mm  |  反演 dx: {final_dx:.3f} mm  |  绝对误差: {abs(TRUE_DX*1e3 - final_dx):.4f} mm")
print(f"真实 dy: {TRUE_DY*1e3:.3f} mm  |  反演 dy: {final_dy:.3f} mm  |  绝对误差: {abs(TRUE_DY*1e3 - final_dy):.4f} mm")
print(f"真实 θ:  {np.degrees(TRUE_THETA):.3f} 度   |  反演 θ:  {final_theta:.3f} 度   |  绝对误差: {abs(np.degrees(TRUE_THETA) - final_theta):.4f} 度")


# ==========================================
# 7. 重建结果的视觉对比与定量数据导出 (严谨的学术闭环)
# ==========================================
print("\n>>> 正在计算定量评价指标并导出高清重建结果...")

# 提取并转换为 Numpy 数组
gt_amp = torch.abs(gt_object).cpu().numpy()
gt_phase = torch.angle(gt_object).cpu().numpy()
recon_amp = torch.abs(recon_object).cpu().numpy()
recon_phase = torch.angle(recon_object).cpu().numpy()

# 自定义 RMSE 与 PSNR 计算
mse_phase = np.mean((gt_phase - recon_phase) ** 2)
rmse_phase = np.sqrt(mse_phase)
mse_amp = np.mean((gt_amp - recon_amp) ** 2)
data_range = gt_amp.max() - gt_amp.min()
psnr_amp = 10 * np.log10((data_range ** 2) / mse_amp) if mse_amp > 1e-10 else float('inf')

print(f">>> 重建振幅 PSNR: {psnr_amp:.2f} dB")
print(f">>> 重建相位 RMSE: {rmse_phase:.4f} rad")

# --- 7.1 绘制并保存重建结果组合图 ---
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
fig2.suptitle("FPM Reconstruction with Rigid Body Error Auto-Calibration", fontsize=16, fontweight='bold')

im0 = axes2[0, 0].imshow(gt_amp, cmap='gray')
axes2[0, 0].set_title("Ground Truth Amplitude", fontsize=14)
fig2.colorbar(im0, ax=axes2[0, 0], fraction=0.046, pad=0.04)

im1 = axes2[0, 1].imshow(gt_phase, cmap='viridis')
axes2[0, 1].set_title("Ground Truth Phase", fontsize=14)
fig2.colorbar(im1, ax=axes2[0, 1], fraction=0.046, pad=0.04)

im2 = axes2[1, 0].imshow(recon_amp, cmap='gray')
axes2[1, 0].set_title(f"Reconstructed Amplitude\nPSNR: {psnr_amp:.2f} dB", fontsize=14)
fig2.colorbar(im2, ax=axes2[1, 0], fraction=0.046, pad=0.04)

im3 = axes2[1, 1].imshow(recon_phase, cmap='viridis')
axes2[1, 1].set_title(f"Reconstructed Phase\nRMSE: {rmse_phase:.4f} rad", fontsize=14)
fig2.colorbar(im3, ax=axes2[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path_recon = "output_simulation/sim_4_4_2_reconstruction_comparison.png"
plt.savefig(save_path_recon, dpi=200)
plt.close(fig2)

# --- 7.2 独立导出高保真切片与 Metrics JSON 数据 ---
recon_save_dir = "output_simulation/sim_4_4_2_results_data"
os.makedirs(recon_save_dir, exist_ok=True)

# 无损保存图像
plt.imsave(os.path.join(recon_save_dir, "recon_amplitude.png"), recon_amp, cmap='gray')
plt.imsave(os.path.join(recon_save_dir, "recon_phase.png"), recon_phase, cmap='viridis')

# 遵循 main.py 的规范，将完整的 metrics 字典保存为 metrics.json
json_path = os.path.join(recon_save_dir, "metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print(f">>> 图像组合图已保存至: {save_path_recon}")
print(f">>> 独立高清切片与完整的 Metrics JSON 数据已汇总至目录: {recon_save_dir}")