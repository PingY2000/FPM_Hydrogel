# inverse.py

from forward import forward_model
import matplotlib.pyplot as plt
from utils import check_range
import torch
from tqdm import tqdm
from jaxtyping import Float, Complex
import os
import numpy as np
import pandas as pd

def compute_k_from_rigid_body(
    led_coords_init: torch.Tensor, # [B, 3] (x, y, z) in meters
    params: torch.Tensor,          # [4] (dx, dy, dz, theta)
    wavelength: float,             # meters
    recon_pixel_size: float,       # meters
    n_medium: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    #params: [shift_x, shift_y, shift_z, rotation_theta]

    # 1. 提取参数
    dx, dy, dz, theta = params[0], params[1], params[2], params[3]
    
    x_init = led_coords_init[:, 0]
    y_init = led_coords_init[:, 1]
    z_init = led_coords_init[:, 2] # 假设 z 是正值 (高度)

    # 2. 应用旋转 (绕 Z 轴, 原点通常为光轴中心)
    # 旋转矩阵: x' = x cos - y sin, y' = x sin + y cos
    x_rot = x_init * torch.cos(theta) - y_init * torch.sin(theta)
    y_rot = x_init * torch.sin(theta) + y_init * torch.cos(theta)
    
    # 3. 应用平移
    x_final = x_rot + dx
    y_final = y_rot + dy
    z_final = z_init + dz # 高度变化会改变入射角
    
    # 4. 计算新的物理 k 向量 (NA = sin(theta))
    # R = sqrt(x^2 + y^2 + z^2)
    R = torch.sqrt(x_final**2 + y_final**2 + z_final**2)
    
    na_x = x_final / R
    na_y = y_final / R
    
    # 5. 归一化到 [-0.5, 0.5] 空间 (cycles per pixel)
    # k = sin(theta) / lambda * pixel_size
    kx_new = (n_medium * na_x / wavelength) * recon_pixel_size
    ky_new = (n_medium * na_y / wavelength) * recon_pixel_size
    
    return kx_new, ky_new
def calculate_k_vectors_from_positions(
    filepath: str,
    lambda_nm: float,
    magnification: float,
    camera_pixel_size_um: float,
    recon_pixel_size_m: float,
    loaded_led_indices: list[int],          
    device: torch.device,                   
    center_led_index: int = 1,
    n_medium: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从LED物理位置CSV文件计算归一化k-vectors，
    并返回筛选后的 led_coords_batch, kx_estimated, ky_estimated
    """
    lambda_m = lambda_nm * 1e-9

    # 1️. 读取 CSV
    df = pd.read_csv(filepath)

    # 2️. 构造 LED 坐标 Tensor
    X_m = torch.tensor(df['X'].values * 1e-3, dtype=torch.float32, device=device)
    Y_m = torch.tensor(df['Y'].values * 1e-3, dtype=torch.float32, device=device)
    Z_m = torch.tensor(df['Z'].values * 1e-3, dtype=torch.float32, device=device)

    all_led_coords = torch.stack([X_m, Y_m, Z_m], dim=1)

    # 3️. 计算 NA
    #na_x = np.sin(np.arctan(X_m / Z_m))
    #na_y = np.sin(np.arctan(Y_m / Z_m))
    R = torch.sqrt(X_m**2 + Y_m**2 + Z_m**2)
    na_x = X_m / R
    na_y = Y_m / R

    # 4️. 归一化 k-vectors
    kx_normalized = (n_medium * na_x / lambda_m) * recon_pixel_size_m
    ky_normalized = (n_medium * na_y / lambda_m) * recon_pixel_size_m

    #print(f"Calculated normalized kx range: [{kx_normalized.min():.3f}, {kx_normalized.max():.3f}]")
    #print(f"Calculated normalized ky range: [{ky_normalized.min():.3f}, {ky_normalized.max():.3f}]")

    # 5️. 根据 loaded_led_indices 进行筛选
    indices_for_slicing = torch.tensor(
        loaded_led_indices,
        dtype=torch.long,
        device=device
    ) - 1   # 从1-based转为0-based

    led_coords_batch = all_led_coords[indices_for_slicing]
    kx = kx_normalized[indices_for_slicing]
    ky = ky_normalized[indices_for_slicing]

    return led_coords_batch, kx, ky

def solve_inverse(
    captures: Float[torch.Tensor, "B n n"],
    object: Complex[torch.Tensor, "N N"],
    pupil: Complex[torch.Tensor, "N N"],
    led_physics_coords: Float[torch.Tensor, "B 3"] | None = None, # [X, Y, Z] 物理坐标(米)
    wavelength: float | None = None,       # 波长(米)
    recon_pixel_size: float | None = None, # 重建像素尺寸(米)
    kx_batch: Float[torch.Tensor, "B"] | None = None, 
    ky_batch: Float[torch.Tensor, "B"] | None = None,
    
    learn_pupil: bool = True,
    learn_k_vectors: bool = False, # 现在这个开关控制是否开启“刚体优化”
    epochs: int = 500,
    vis_interval: int = 0,
) -> tuple[Complex[torch.Tensor, "N N"], Complex[torch.Tensor, "N N"], Float[torch.Tensor, "B"], Float[torch.Tensor, "B"], dict[str, list[float]]]:

    #check_range(captures, 0, 1, "captures")
    #check_range(object, 0, 1, "object")
    #check_range(pupil, 0, 1, "pupil")
    #check_range(kx_batch, -0.5, 0.5, "kx_batch")
    #check_range(ky_batch, -0.5, 0.5, "ky_batch")


    output_size = object.shape[0]
    downsample_factor = output_size // captures[0].shape[0]
    device = object.device
    
    # --- 1. 参数校验与初始化 ---
    use_rigid_body = learn_k_vectors and (led_physics_coords is not None)
    
    if use_rigid_body:
        print(">>> 启用 K-Vector 校准 (Global Shift & Rotation)")
        if wavelength is None or recon_pixel_size is None:
            raise ValueError("需要提供 wavelength 和 recon_pixel_size")
        led_physics_coords = led_physics_coords.to(device)
        rigid_params = torch.zeros(4, device=device, requires_grad=True)
        curr_kx, curr_ky = compute_k_from_rigid_body(led_physics_coords, rigid_params, wavelength, recon_pixel_size)
    else:
        # 回退到旧模式 (直接使用传入的 kx, ky)
        if kx_batch is None or ky_batch is None:
            raise ValueError("须提供 kx_batch 和 ky_batch")
        curr_kx = kx_batch.detach().clone()
        curr_ky = ky_batch.detach().clone()
        if learn_k_vectors: # 旧的单点独立优化模式
             print(">>> 启用 K-Vector 校准 ")
             curr_kx.requires_grad_(True)
             curr_ky.requires_grad_(True)

    # --- 2. 优化器设置 ---
    learned_tensors = []
    base_max_lrs = []  # 用于动态存储每个组的最大学习率

    # 1. Object 参数
    object = object.clone().detach().requires_grad_(True)
    learned_tensors.append({'params': object})
    base_max_lrs.append(0.1) 

    # 2. Pupil 参数 (动态添加)
    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': pupil})
        base_max_lrs.append(0.1)

    # 3. 刚体变换或 K 向量参数 (动态添加)
    if use_rigid_body:
        # 刚体参数 [dx, dy, dz, theta]
        learned_tensors.append({'params': rigid_params})
        base_max_lrs.append(1e-3) # 这里设为 1e-3，OneCycle 会从约 4e-5 开始增加
    elif learn_k_vectors:
        # 旧模式：独立优化每个 LED 的 kx, ky
        learned_tensors.append({'params': curr_kx})
        base_max_lrs.append(1e-2)
        learned_tensors.append({'params': curr_ky})
        base_max_lrs.append(1e-2)

    optimizer = torch.optim.AdamW(learned_tensors)

    # 初始化调度器：使用动态生成的 base_max_lrs 列表
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=base_max_lrs, 
        total_steps=epochs, 
        pct_start=0.3
    )
    
    metrics = {'loss': [], 'lr': [], 'dx': [], 'dy': [], 'dz': [], 'theta': []}
    snapshots = [] # 用于存储可视化数据

    # 记录初始 K 向量用于对比
    with torch.no_grad():
        if use_rigid_body:
            init_kx, init_ky = compute_k_from_rigid_body(
                led_physics_coords, torch.zeros(4, device=device), wavelength, recon_pixel_size
            )
        else:
            init_kx, init_ky = kx_batch, ky_batch
    
    #创建进度条对象
    loop = tqdm(range(epochs), desc="Solving")
    
    # Training loop
    for _ in loop:
        if use_rigid_body:
            if _ > 200: 
                # 允许梯度更新
                rigid_params.requires_grad_(True)
                curr_kx, curr_ky = compute_k_from_rigid_body(
                    led_physics_coords, rigid_params, wavelength, recon_pixel_size
                )
            else:
                # 锁定坐标，仅使用初始值
                if use_rigid_body:
                    rigid_params.requires_grad_(False)
                    curr_kx, curr_ky = compute_k_from_rigid_body(
                        led_physics_coords, torch.zeros_like(rigid_params), wavelength, recon_pixel_size
                    )
            # 记录参数变化
            metrics['dx'].append(rigid_params[0].item())
            metrics['dy'].append(rigid_params[1].item())
            metrics['dz'].append(rigid_params[2].item())
            metrics['theta'].append(rigid_params[3].item())
        # Batched forward pass
        predicted_intensities = forward_model(object, pupil, curr_kx, curr_ky, downsample_factor)

        # Compute loss across all captures
        loss = torch.nn.functional.l1_loss(predicted_intensities, captures)
        '''
        if use_rigid_body:
            # 物理约束：惩罚偏离原点。将物理量除以它们的典型偏差范围，再平方
            scale_d = 1e-3  # 毫米级别的容忍度
            scale_theta = 0.05 # 约 3 度的容忍度
            reg_loss = ( (rigid_params[0]/scale_d)**2 + 
                        (rigid_params[1]/scale_d)**2 + 
                        (rigid_params[2]/scale_d)**2 + 
                        (rigid_params[3]/scale_theta)**2 ) * 0.1 # 控制整体权重
            #reg_loss = torch.sum(rigid_params**2) * 1e5 
            loss = loss + reg_loss
        '''
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['loss'].append(loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

        # --- 过程数据采样 ---
        if vis_interval > 0 and (_ % vis_interval == 0 or _ == epochs - 1):
            with torch.no_grad():
                snap = {
                    'epoch': _,
                    'object': object.detach().cpu().numpy().copy(),
                    'pupil': pupil.detach().cpu().numpy().copy(),
                    'kx': curr_kx.detach().cpu().numpy().copy(),
                    'ky': curr_ky.detach().cpu().numpy().copy(),
                }
                if use_rigid_body:
                    snap['rigid_params'] = rigid_params.detach().cpu().numpy().copy()
                snapshots.append(snap)

    return object.detach(), pupil.detach(), curr_kx.detach(), curr_ky.detach(), metrics, snapshots, init_kx, init_ky

def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    #计算 3D 复合张量的全变分 (Total Variation)
    # 空间维度 TV (X 和 Y 方向)
    tv_h = torch.sum(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
    tv_w = torch.sum(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    # 深度维度 TV (Z 方向) - 使得相邻切片之间过渡平滑
    tv_d = torch.sum(torch.abs(x[1:, :, :] - x[:-1, :, :])) if x.shape[0] > 1 else 0.0
    return tv_h + tv_w + tv_d

