# ptych\inverse.py
from ptych.forward import forward_model
import matplotlib.pyplot as plt
#from ptych.utils import check_range
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
    recon_pixel_size: float        # meters
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据刚体变换参数实时计算 kx, ky
    params: [shift_x, shift_y, shift_z, rotation_theta]
    """
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
    kx_new = (na_x / wavelength) * recon_pixel_size
    ky_new = (na_y / wavelength) * recon_pixel_size
    
    return kx_new, ky_new

def calculate_k_vectors_from_positions(
    X_m: np.ndarray,
    Y_m: np.ndarray,
    Z_m: np.ndarray,
    lambda_nm: float,
    magnification: float,
    camera_pixel_size_um: float,
    recon_pixel_size_m: float,
    center_led_index: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从LED物理位置CSV文件计算归一化的k-vectors。"""
    lambda_m = lambda_nm * 1e-9
    
    # --- 计算 NA ---
    # 使用独立计算，这在大多数情况下足够准确
    na_x = np.sin(np.arctan(X_m / Z_m))
    na_y = np.sin(np.arctan(Y_m / Z_m))

    # --- 转换为归一化 k-vectors ---
    # k_normalized 的单位是 "cycles per pixel"
    # 它代表了由该NA在样品平面上，每个像素经历的相位周期数
    kx_normalized = na_x / lambda_m * recon_pixel_size_m
    ky_normalized = na_y / lambda_m * recon_pixel_size_m

    print(f"Calculated normalized kx range: [{kx_normalized.min():.3f}, {kx_normalized.max():.3f}]")
    print(f"Calculated normalized ky range: [{ky_normalized.min():.3f}, {ky_normalized.max():.3f}]")

    return torch.from_numpy(kx_normalized).float(), torch.from_numpy(ky_normalized).float()

def solve_inverse(
    captures: Float[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
    object: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    pupil: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
    epochs: int = 500,
    vis_interval: int = 0,
) -> tuple[Complex[torch.Tensor, "N N"],
    Complex[torch.Tensor, "N N"],
    Float[torch.Tensor, "B"],
    Float[torch.Tensor, "B"],
    dict[str, list[float]]]:

    #check_range(captures, 0, 1, "captures")
    #check_range(object, 0, 1, "object")
    #check_range(pupil, 0, 1, "pupil")
    #check_range(kx_batch, -0.5, 0.5, "kx_batch")
    #check_range(ky_batch, -0.5, 0.5, "ky_batch")


    output_size = object.shape[0]
    downsample_factor = output_size // captures[0].shape[0]
    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)
    print("Downsample factor:", downsample_factor)

    learned_tensors: list[dict[str, torch.Tensor | float]] = []
    object = object.clone().detach().requires_grad_(True)
    learned_tensors.append({'params': object, 'lr': 0.1})

    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': pupil, 'lr': 0.1})

    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': kx_batch, 'lr': 0.1})
        learned_tensors.append({'params': ky_batch, 'lr': 0.1})

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(learned_tensors)

    # Add scheduler
    """scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # total epochs
        eta_min=0.01  # minimum LR
    )"""

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.05,
        total_steps=epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1e4,
    )

    # Telemetry
    metrics: dict[str, list[float]] = {
        'loss': [],
        'lr': []
    }




    if vis_interval:
        # --- 准备实时显示画布 ---
        # --- 1. 初始化和准备保存路径 ---
        os.makedirs("output", exist_ok=True)
        # 计算需要展示的快照数量 (包括最后一帧)
        snapshot_indices = list(range(0, epochs, vis_interval))
        if (epochs - 1) not in snapshot_indices:
            snapshot_indices.append(epochs - 1)
        
        num_snapshots = len(snapshot_indices)
        
        # 创建大图：每一行代表一次采样，列为 [幅值, 相位]
        # 增加 figsize 以确保高密度下依然清晰
        fig, axes = plt.subplots(num_snapshots, 4, figsize=(16, 4 * num_snapshots)) # 增加宽度
        plt.subplots_adjust(hspace=0.3, wspace=0.2) 
        
        snapshot_count = 0

    
    # Training loop
    loop = tqdm(range(epochs), desc="Solving")
    for _ in loop:
        # Batched forward pass
        predicted_intensities = forward_model(object, pupil, kx_batch, ky_batch, downsample_factor)  # [B, H, W]

        # Compute loss across all captures
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        current_loss = total_loss.item()
        metrics['loss'].append(current_loss)
        metrics['lr'].append(scheduler.get_last_lr()[0])


        if vis_interval:
            if _ in snapshot_indices:
                with torch.no_grad():
                    # --- 获取 Object 数据 ---
                    obj_np = object.detach().cpu()
                    obj_amp = torch.abs(obj_np).numpy()
                    obj_phase = torch.angle(obj_np).numpy()

                    # --- 获取 Pupil 数据 ---
                    pup_np = pupil.detach().cpu()
                    pup_amp = torch.abs(pup_np).numpy()
                    pup_phase = torch.angle(pup_np).numpy()

                    # --- 绘制到对应的列 ---
                    # 第一列：Object 幅值
                    axes[snapshot_count, 0].imshow(obj_amp, cmap='gray')
                    axes[snapshot_count, 0].set_title(f"Ep {_} | Obj Amp")
                    
                    # 第二列：Object 相位
                    axes[snapshot_count, 1].imshow(obj_phase, cmap='viridis')
                    axes[snapshot_count, 1].set_title(f"Ep {_} | Obj Phase")
                    
                    # 第三列：Pupil 幅值
                    axes[snapshot_count, 2].imshow(pup_amp, cmap='gray')
                    axes[snapshot_count, 2].set_title(f"Ep {_} | Pup Amp")
                    
                    # 第四列：Pupil 相位
                    axes[snapshot_count, 3].imshow(pup_phase, cmap='magma') # 使用不同色阶区分
                    axes[snapshot_count, 3].set_title(f"Ep {_} | Pup Phase")

                    # 统一关闭坐标轴
                    for col in range(4):
                        axes[snapshot_count, col].axis('off')

                    snapshot_count += 1



    if vis_interval:
        # --- 4. 保存最终的高密度大图 ---
        save_path = "output/iteration_process_dense.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # 释放内存，不显示窗口
        
        print(f"\nIteration progress saved to: {save_path}")

    # --- Detach learned parameters ---
    final_object = object.detach()
    final_pupil = pupil.detach()

    if learn_k_vectors:
        final_kx = kx_batch.detach()
        final_ky = ky_batch.detach()
    else:
        # 若未学习，则返回原始值（保持接口一致）
        final_kx = kx_batch
        final_ky = ky_batch

    return final_object, final_pupil, final_kx, final_ky, metrics

