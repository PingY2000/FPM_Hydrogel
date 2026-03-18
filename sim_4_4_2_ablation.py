# sim_4_4_2_ablation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

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
LED_POSITIONS_FILE = r"LedPosHis\led_positions 0210.csv" # 确保路径正确

original_pixel_size_m = CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION
recon_pixel_size_m = original_pixel_size_m / DOWNSAMPLE_FACTOR
N_RECON = 512 # 尺寸适中，确保单次运行时间可控

device = get_default_device()
torch.set_default_device(device)
print(f"[{device}] 初始化 4.4.2 误差校正消融实验 (Ablation Study)...")

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

gt_amp = torch.abs(gt_object).cpu().numpy()
gt_phase = torch.angle(gt_object).cpu().numpy()

# ==========================================
# 3. 注入已知的物理刚体误差
# ==========================================
df = pd.read_csv(LED_POSITIONS_FILE)
loaded_led_indices = list(range(1, len(df) + 1))
led_coords_ideal, kx_ideal, ky_ideal = calculate_k_vectors_from_positions(
    filepath=LED_POSITIONS_FILE, lambda_nm=WAVELENGTH_M * 1e9,
    magnification=MAGNIFICATION, camera_pixel_size_um=CAMERA_PIXEL_SIZE_UM,
    recon_pixel_size_m=recon_pixel_size_m, loaded_led_indices=loaded_led_indices,
    device=device, center_led_index=1
)

# 注入工程公差误差
TRUE_DX = 0.3e-3         
TRUE_DY = -0.8e-3        
TRUE_DZ = 0.0
TRUE_THETA = np.radians(0.4) 

true_rigid_params = torch.tensor([TRUE_DX, TRUE_DY, TRUE_DZ, TRUE_THETA], device=device)
kx_corrupted, ky_corrupted = compute_k_from_rigid_body(
    led_coords_ideal, true_rigid_params, WAVELENGTH_M, recon_pixel_size_m
)

pupil_radius_pixels = (NA_OBJECTIVE / WAVELENGTH_M) * recon_pixel_size_m * N_RECON
gt_pupil = create_circular_pupil((N_RECON, N_RECON), radius=int(pupil_radius_pixels)).to(device)

print("生成带有误差的低分辨率原图...")
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

# 初始猜测
center_idx = torch.argmin(kx_ideal**2 + ky_ideal**2)
obj_guess_amp = F.interpolate(simulated_captures[center_idx:center_idx+1].unsqueeze(0), size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)

EPOCHS = 400 # 统一迭代次数以公平对比时间

# 定量指标计算函数
def calc_metrics(recon_obj):
    r_amp = torch.abs(recon_obj).cpu().numpy()
    r_phase = torch.angle(recon_obj).cpu().numpy()
    rmse = np.sqrt(np.mean((gt_phase - r_phase) ** 2))
    mse_amp = np.mean((gt_amp - r_amp) ** 2)
    psnr = 10 * np.log10(((gt_amp.max() - gt_amp.min()) ** 2) / mse_amp) if mse_amp > 1e-10 else float('inf')
    return r_amp, r_phase, psnr, rmse

results = {}

# ==========================================
# 实验 A: 不开启误差校正 (Baseline)
# ==========================================
print("\n>>> [实验 A] 不开启误差校正 (learn_k_vectors=False) ...")
t0 = time.time()
recon_A, _, _, _, _, _, _, _ = solve_inverse(
    captures=simulated_captures, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=None, wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=False, learn_k_vectors=False, epochs=EPOCHS, vis_interval=0
)
time_A = time.time() - t0
amp_A, phase_A, psnr_A, rmse_A = calc_metrics(recon_A)
results['None'] = {'amp': amp_A, 'phase': phase_A, 'psnr': psnr_A, 'rmse': rmse_A, 'time': time_A}

# ==========================================
# 实验 B: 独立 K 向量优化 (非刚体)
# ==========================================
print("\n>>> [实验 B] 开启非刚体校正 (learn_k_vectors=True, 无物理坐标约束) ...")
t0 = time.time()
recon_B, _, _, _, _, _, _, _ = solve_inverse(
    captures=simulated_captures, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=None,  # 传入 None，触发独立 kx, ky 优化
    wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=False, learn_k_vectors=True, epochs=EPOCHS, vis_interval=0
)
time_B = time.time() - t0
amp_B, phase_B, psnr_B, rmse_B = calc_metrics(recon_B)
results['Non-Rigid'] = {'amp': amp_B, 'phase': phase_B, 'psnr': psnr_B, 'rmse': rmse_B, 'time': time_B}

# ==========================================
# 实验 C: 四参数刚体联合优化 (Proposed)
# ==========================================
print("\n>>> [实验 C] 开启四参数刚体校正 (learn_k_vectors=True, 有物理坐标约束) ...")
t0 = time.time()
recon_C, _, _, _, _, _, _, _ = solve_inverse(
    captures=simulated_captures, object=obj_guess, pupil=gt_pupil, 
    led_physics_coords=led_coords_ideal, # 传入真实物理坐标，触发刚体优化
    wavelength=WAVELENGTH_M, recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_ideal, ky_batch=ky_ideal,
    learn_pupil=False, learn_k_vectors=True, epochs=EPOCHS, vis_interval=0
)
time_C = time.time() - t0
amp_C, phase_C, psnr_C, rmse_C = calc_metrics(recon_C)
results['Rigid-Body'] = {'amp': amp_C, 'phase': phase_C, 'psnr': psnr_C, 'rmse': rmse_C, 'time': time_C}

# ==========================================
# 4. 汇总与绘图 (生成学术消融对比图)
# ==========================================
print("\n================ 实验结果汇总 ================")
print(f"A. 无校正      | 时间: {time_A:.2f}s | PSNR: {psnr_A:.2f} dB | RMSE: {rmse_A:.4f} rad")
print(f"B. 非刚体校正  | 时间: {time_B:.2f}s | PSNR: {psnr_B:.2f} dB | RMSE: {rmse_B:.4f} rad")
print(f"C. 刚体校正    | 时间: {time_C:.2f}s | PSNR: {psnr_C:.2f} dB | RMSE: {rmse_C:.4f} rad")
print("==============================================")

os.makedirs("output_simulation/ablation_study", exist_ok=True)

fig, axes = plt.subplots(3, 2, figsize=(10, 14))
fig.suptitle("Ablation Study: Illumination Error Correction Strategies", fontsize=16, fontweight='bold', y=0.98)

methods = ['None', 'Non-Rigid', 'Rigid-Body']
titles = ['(a) No Correction', '(b) Non-Rigid Optimization', '(c) Proposed Rigid-Body']

for i, method in enumerate(methods):
    res = results[method]
    
    # Amplitude
    im_a = axes[i, 0].imshow(res['amp'], cmap='gray')
    axes[i, 0].set_title(f"{titles[i]} - Amp\nPSNR: {res['psnr']:.2f} dB | Time: {res['time']:.1f} s", fontsize=12)
    axes[i, 0].axis('off')
    fig.colorbar(im_a, ax=axes[i, 0], fraction=0.046, pad=0.04)
    
    # Phase
    im_p = axes[i, 1].imshow(res['phase'], cmap='viridis')
    axes[i, 1].set_title(f"{titles[i]} - Phase\nRMSE: {res['rmse']:.4f} rad", fontsize=12)
    axes[i, 1].axis('off')
    fig.colorbar(im_p, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = "output_simulation/ablation_study/correction_ablation_comparison.png"
plt.savefig(save_path, dpi=200)
print(f"\n>>> 完美！消融实验对比大图已保存至: {save_path}")
plt.show()

# ==========================================
# 5. 独立导出三种情况的高保真切片原图 (供高清排版)
# ==========================================
print("\n>>> 正在独立保存三种实验策略的高清重建原图...")
export_dir = "output_simulation/ablation_study/raw_reconstructions"
os.makedirs(export_dir, exist_ok=True)

# 映射文件名，避免出现特殊字符
method_names_map = {
    'None': '1_no_correction',
    'Non-Rigid': '2_non_rigid',
    'Rigid-Body': '3_rigid_body'
}

for method_key, file_prefix in method_names_map.items():
    res = results[method_key]
    
    # 导出振幅 (灰度)
    plt.imsave(
        os.path.join(export_dir, f"{file_prefix}_amplitude.png"), 
        res['amp'], 
        cmap='gray'
    )
    # 导出相位 (伪彩)
    plt.imsave(
        os.path.join(export_dir, f"{file_prefix}_phase.png"), 
        res['phase'], 
        cmap='viridis'
    )

# 顺便导出 Ground Truth 作为绝对参照组
plt.imsave(os.path.join(export_dir, "0_ground_truth_amplitude.png"), gt_amp, cmap='gray')
plt.imsave(os.path.join(export_dir, "0_ground_truth_phase.png"), gt_phase, cmap='viridis')

print(f">>> 完美！共计 8 张高清原图已分类保存至: {export_dir}")