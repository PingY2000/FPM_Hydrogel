# sim_4_4_1_baseline.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# 导入你现有的模块
from forward import forward_model
from inverse import solve_inverse, calculate_k_vectors_from_positions
from utils import get_default_device, create_circular_pupil

# ==========================================
# 1. 仿真系统参数配置 (严格对标你的实际物理参数)
# ==========================================
WAVELENGTH_M = 525e-9          # 波长 525 nm
NA_OBJECTIVE = 0.1            # 物镜 NA
MAGNIFICATION = 4.0           # 放大倍率
CAMERA_PIXEL_SIZE_UM = 3.45    # 相机像元
DOWNSAMPLE_FACTOR = 1          # 仿真降采样因子 (验证分辨率突破)

# CSV 配置文件路径
LED_POSITIONS_FILE = "LedPosHis\led_positions 0210.csv"

# 计算物理像素尺寸
original_pixel_size_m = CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION
recon_pixel_size_m = original_pixel_size_m / DOWNSAMPLE_FACTOR

N_RECON = 512 # 重建的高分辨率尺寸 (为保证仿真速度，设定为 256x256)

device = get_default_device()
torch.set_default_device(device)
print(f"[{device}] 初始化 4.4.1 理想薄物体基准仿真...")

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
# 3. 从 CSV 加载照明 K 向量与生成光瞳
# ==========================================
if not os.path.exists(LED_POSITIONS_FILE):
    raise FileNotFoundError(f"找不到 LED 坐标文件: {LED_POSITIONS_FILE}")

# 读取 CSV 行数以确定 LED 总数
df = pd.read_csv(LED_POSITIONS_FILE)
total_leds = len(df)
loaded_led_indices = list(range(1, total_leds + 1)) # 模拟加载 CSV 中所有的 LED

print(f"从 CSV 中读取到 {total_leds} 个 LED 位置，正在计算 K 向量...")

# 调用你的核心计算函数
led_coords_batch, kx_batch, ky_batch = calculate_k_vectors_from_positions(
    filepath=LED_POSITIONS_FILE,
    lambda_nm=WAVELENGTH_M * 1e9, # 转换为 nm
    magnification=MAGNIFICATION,
    camera_pixel_size_um=CAMERA_PIXEL_SIZE_UM,
    recon_pixel_size_m=recon_pixel_size_m,
    loaded_led_indices=loaded_led_indices,
    device=device,
    center_led_index=1 # 假设第一颗灯是中心照明
)

# 计算光瞳半径 (像素)
pupil_radius_pixels = (NA_OBJECTIVE / WAVELENGTH_M) * recon_pixel_size_m * N_RECON
gt_pupil = create_circular_pupil((N_RECON, N_RECON), radius=int(pupil_radius_pixels)).to(device)

# ==========================================
# 4. 正向物理过程：模拟生成低分辨率图
# ==========================================
print("执行前向物理模型模拟低分辨率拍摄...")
with torch.no_grad():
    simulated_captures = forward_model(
        gt_object, gt_pupil, kx_batch, ky_batch, downsample_factor=DOWNSAMPLE_FACTOR
    )
    # 添加极其微弱的本底噪声 (35dB) 以避免除零错误
    for i in range(len(kx_batch)):
        local_max = simulated_captures[i].max()
        noise_level = local_max * (10 ** (-35 / 20))
        simulated_captures[i] += torch.randn_like(simulated_captures[i]) * noise_level
        
    simulated_captures = torch.clamp(simulated_captures, min=0)
    simulated_captures += torch.randn_like(simulated_captures) * noise_level
    simulated_captures = torch.clamp(simulated_captures, min=0)

save_dir = "output_simulation/simulated_captures"
os.makedirs(save_dir, exist_ok=True)

print(f"正在将 {len(simulated_captures)} 张模拟低分辨率图导出到本地...")
captures_np = simulated_captures.cpu().numpy()

for i in range(len(captures_np)):
    img = captures_np[i]
    
    # 【独立归一化】：为了让暗场的高频边缘也能被肉眼看清，
    # 我们将每张图的对比度独立拉伸到 [0, 1] 区间
    img_min = img.min()
    img_max = img.max()
    img_normalized = (img - img_min) / (img_max - img_min + 1e-8)
    
    
    # 命名格式：索引_kx_ky.png
    filename = f"capture_{i:03d}.png"
    filepath = os.path.join(save_dir, filename)
    
    # 保存为灰度 PNG
    plt.imsave(filepath, img_normalized, cmap='gray')

print(f">>> 导出完成！请前往 {save_dir} 文件夹查看。")

# ==========================================
# 5. 逆向重建过程
# ==========================================
print("执行逆向重建算法...")
# 自动寻找距离中心最近的 LED 作为初始振幅猜测
center_idx = torch.argmin(kx_batch**2 + ky_batch**2)
center_img = simulated_captures[center_idx:center_idx+1].unsqueeze(0)

obj_guess_amp = F.interpolate(center_img, size=(N_RECON, N_RECON), mode='bicubic')[0, 0]
obj_guess = torch.complex(obj_guess_amp, torch.zeros_like(obj_guess_amp)).to(device)
pupil_guess = create_circular_pupil((N_RECON, N_RECON), radius=int(pupil_radius_pixels)).to(device)

recon_object, recon_pupil, _, _, metrics, _, _, _ = solve_inverse(
    captures=simulated_captures,
    object=obj_guess,
    pupil=pupil_guess,
    led_physics_coords=None,
    wavelength=WAVELENGTH_M,
    recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_batch,
    ky_batch=ky_batch,
    learn_pupil=False,        
    learn_k_vectors=False,    
    epochs=200,               
    vis_interval=0
)

# ==========================================
# 6. 定量误差评估与结果可视化 (纯 Numpy 实现)
# ==========================================
print("重建完成，正在计算客观评价指标...")
gt_amp = torch.abs(gt_object).cpu().numpy()
gt_phase = torch.angle(gt_object).cpu().numpy()
recon_amp = torch.abs(recon_object).cpu().numpy()
recon_phase = torch.angle(recon_object).cpu().numpy()

# 自定义 RMSE 计算 (相位)
mse_phase = np.mean((gt_phase - recon_phase) ** 2)
rmse_phase = np.sqrt(mse_phase)

# 自定义 PSNR 计算 (振幅)
mse_amp = np.mean((gt_amp - recon_amp) ** 2)
data_range = gt_amp.max() - gt_amp.min()
psnr_amp = 10 * np.log10((data_range ** 2) / mse_amp) if mse_amp > 1e-10 else float('inf')

print(f">>> 振幅 PSNR: {psnr_amp:.2f} dB")
print(f">>> 相位 RMSE: {rmse_phase:.4f} rad")

# 绘图输出
os.makedirs("output_simulation", exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 行1：Ground Truth
im0 = axes[1, 0].imshow(simulated_captures[center_idx].cpu().numpy(), cmap='gray')
axes[1, 0].set_title(f"Center Low-Res Capture\n(NA: {NA_OBJECTIVE})", fontsize=12)

im1 = axes[0, 1].imshow(gt_amp, cmap='gray')
axes[0, 1].set_title("Ground Truth Amplitude", fontsize=12)
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[0, 2].imshow(gt_phase, cmap='viridis')
axes[0, 2].set_title("Ground Truth Phase", fontsize=12)
fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# 行2：Reconstruction
axes[0, 0].plot(metrics['loss'], color='red', linewidth=2)
axes[0, 0].set_title("Convergence Curve (L1 Loss)", fontsize=12)
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].grid(True, linestyle='--')

im3 = axes[1, 1].imshow(recon_amp, cmap='gray')
axes[1, 1].set_title(f"Reconstructed Amplitude\nPSNR: {psnr_amp:.2f} dB", fontsize=12)
fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

im4 = axes[1, 2].imshow(recon_phase, cmap='viridis')
axes[1, 2].set_title(f"Reconstructed Phase\nRMSE: {rmse_phase:.4f} rad", fontsize=12)
fig.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = "output_simulation/sim_4_4_1_baseline_result.png"
plt.savefig(save_path, dpi=200)
print(f"验证完成！基准仿真结果大图已保存至: {save_path}")
plt.show()

# ==========================================
# 7. 独立导出高保真重构振幅与相位图 (供论文高清排版使用)
# ==========================================
recon_save_dir = "output_simulation/sim_4_4_1_reconstruction_results"
os.makedirs(recon_save_dir, exist_ok=True)

print("正在单独导出重建与基准图像的高清切片...")

# 1. 导出重建振幅与相位
# imsave 会自动将数据的极值映射到色表的两端，保证极佳的视觉对比度
plt.imsave(
    os.path.join(recon_save_dir, "reconstructed_amplitude.png"), 
    recon_amp, 
    cmap='gray'
)
plt.imsave(
    os.path.join(recon_save_dir, "reconstructed_phase.png"), 
    recon_phase, 
    cmap='viridis'  # 保持一致的学术伪彩色表
)

# 2. 导出 Ground Truth 振幅与相位 (作为完美的对照组)
plt.imsave(
    os.path.join(recon_save_dir, "ground_truth_amplitude.png"), 
    gt_amp, 
    cmap='gray'
)
plt.imsave(
    os.path.join(recon_save_dir, "ground_truth_phase.png"), 
    gt_phase, 
    cmap='viridis'
)

# 3. 导出中心低分辨率原图 (用于展示物镜的原始截止频率)
plt.imsave(
    os.path.join(recon_save_dir, "center_low_res_capture.png"), 
    simulated_captures[center_idx].cpu().numpy(), 
    cmap='gray'
)

print(f">>> 独立的重建图片已全部无损保存至: {recon_save_dir} 文件夹中。")


# 遵循 main.py 的规范，将完整的 metrics 字典保存为 metrics.json
json_path = os.path.join(recon_save_dir, "metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f">>> Metrics JSON 数据已汇总至目录: {recon_save_dir}")
