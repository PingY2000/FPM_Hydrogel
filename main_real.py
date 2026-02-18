# main_real.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import ptych
import os
import json
import pandas as pd

from ptych import solve_inverse, analysis, calculate_k_vectors_from_positions, compute_k_from_rigid_body
from utils import (
    get_default_device,
    load_real_captures,
    create_circular_pupil,
)
from visualize import(
    visualize_kspace_and_captures,
    visualize_reconstruction,
    visualize_pupil,
    save_training_metrics
)

# --- 1. 用户配置 (User Configuration) ---
# ----------------------------------------------------

# 定义测试集配置文件路径
CONFIG_JSON_PATH = r"D:\FPM_Dataset\OnTest\param.json"

if not os.path.exists(CONFIG_JSON_PATH):
    raise FileNotFoundError(f"找不到配置文件: {CONFIG_JSON_PATH}")
with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# A. 从 JSON 中读取系统物理参数
NA_OBJECTIVE = config_data.get('NA_OBJECTIVE', 0.5)          # 物镜 NA
WAVELENGTH_NM = config_data.get('WAVELENGTH_NM', 525)        # LED 波长 (nm)
MAGNIFICATION = config_data.get('MAGNIFICATION', 10.0)       # 物镜放大倍率
CAMERA_PIXEL_SIZE_UM = config_data.get('CAMERA_PIXEL_SIZE_UM', 3.45) # 相机像素尺寸 (um)

#print(f"--- 系统参数已加载 ---")
print(f"NA: {NA_OBJECTIVE}\nWavelength: {WAVELENGTH_NM}nm\nMag: {MAGNIFICATION}x\nAMERA_PIXEL_SIZE:{CAMERA_PIXEL_SIZE_UM}\n")

# B. 数据和重建参数
CAPTURES_PATH = "D:\FPM_Dataset\OnTest\TIF" # 原始图片文件目录
LED_POSITIONS_FILE = "led_positions.csv" # LED位置文件
CENTER_LED_INDEX = 1        # 对应中心照明的LED的索引号 (从1开始)
DOWNSAMPLE_FACTOR = 2       # 生成图片分辨率倍数

LEARN_PUPIL = True # 校正像差
LEARN_K_VECTORS = True # 修正k-vector误差
USE_RIGID_BODY= True # 启动刚体校准
EPOCHS = 200 # Epochs 上限
VIS_INTERVAL = 20 # 迭代过程图片展示间隔




# --- 2. 设备设置 ---
# ----------------------------------------------------
pytorch_device = get_default_device()
torch.set_default_device(pytorch_device)
print(f"Running on device: {pytorch_device}")


# --- 3. 加载并预处理原始图片 ---
# ----------------------------------------------------
captures, loaded_led_indices = load_real_captures(CAPTURES_PATH, file_pattern="*.tif")
captures = captures.to(pytorch_device)
#captures = captures / captures.mean(dim=(-1, -2), keepdim=True)


# --- 4. 计算和创建初始猜测 ---
# ----------------------------------------------------
# A. 确定重建尺寸
capture_height, capture_width = captures.shape[-2:]
output_size = capture_height * DOWNSAMPLE_FACTOR # 最终重建图像尺寸
print(f"Capture size: {capture_height}x{capture_width}, Output size: {output_size}x{output_size}")

# B. 计算物理常数
recon_pixel_size_m = (CAMERA_PIXEL_SIZE_UM * 1e-6 / MAGNIFICATION) / DOWNSAMPLE_FACTOR
wavelength_val = WAVELENGTH_NM * 1e-9

# C. 计算K向量 
led_coords_batch, kx_estimated, ky_estimated = \
    calculate_k_vectors_from_positions(
        filepath=LED_POSITIONS_FILE,
        lambda_nm=WAVELENGTH_NM,
        magnification=MAGNIFICATION,
        camera_pixel_size_um=CAMERA_PIXEL_SIZE_UM,
        recon_pixel_size_m=recon_pixel_size_m,
        loaded_led_indices=loaded_led_indices,
        device=pytorch_device,
        center_led_index=CENTER_LED_INDEX
    )
# 角度可视化调用 
visualize_kspace_and_captures(
    captures=captures,
    kx_normalized=kx_estimated,
    ky_normalized=ky_estimated
)

# D. 创建光瞳初始猜测
pupil_radius_pixels = (NA_OBJECTIVE / (WAVELENGTH_NM * 1e-9)) / recon_pixel_size_m * output_size
print(f"Calculated initial pupil radius: {pupil_radius_pixels:.1f} pixels")
pupil_guess = create_circular_pupil((output_size, output_size), radius=int(pupil_radius_pixels))

# E. 创建物体初始猜测
#object_guess = 0.5 * torch.ones(int(output_size), int(output_size), dtype=torch.complex64)
# 更好的初始化（假设 captures[0] 是中心照明）
import torch.nn.functional as F
object_guess = F.interpolate(captures[0:1, None, :, :], 
                             size=(output_size, output_size), 
                             mode='bicubic')[0, 0].to(torch.complex64)

# --- 5. 运行FPM重建 ---
# ----------------------------------------------------
print("\nStarting FPM reconstruction on real data....")
reconstructed_object, reconstructed_pupil, learned_kx, learned_ky, metrics = solve_inverse(
    captures=captures,
    object=object_guess,
    pupil=pupil_guess,
    led_physics_coords=led_coords_batch if USE_RIGID_BODY else None,
    wavelength=wavelength_val,
    recon_pixel_size=recon_pixel_size_m,
    kx_batch=kx_estimated, 
    ky_batch=ky_estimated, 
    learn_pupil=LEARN_PUPIL,       
    learn_k_vectors=LEARN_K_VECTORS, 
    epochs=EPOCHS, 
    vis_interval=VIS_INTERVAL
)
print("Reconstruction finished.")

# --- 6. 可视化结果 & 保存结果 ---
save_training_metrics(metrics)
visualize_reconstruction(reconstructed_object)
visualize_pupil(reconstructed_pupil)

print("\nAll plots saved to 'output' folder.")

