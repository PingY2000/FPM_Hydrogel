# inverse.py
from forward import forward_model, forward_model_multislice
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

    # -------------------------
    # 1️⃣ 读取 CSV
    # -------------------------
    df = pd.read_csv(filepath)

    # -------------------------
    # 2️⃣ 构造 LED 坐标 Tensor
    # -------------------------
    X_m = torch.tensor(df['X'].values * 1e-3, dtype=torch.float32, device=device)
    Y_m = torch.tensor(df['Y'].values * 1e-3, dtype=torch.float32, device=device)
    Z_m = torch.tensor(df['Z'].values * 1e-3, dtype=torch.float32, device=device)

    all_led_coords = torch.stack([X_m, Y_m, Z_m], dim=1)

    # -------------------------
    # 3️⃣ 计算 NA
    # -------------------------
    #na_x = np.sin(np.arctan(X_m / Z_m))
    #na_y = np.sin(np.arctan(Y_m / Z_m))
    R = torch.sqrt(X_m**2 + Y_m**2 + Z_m**2)
    na_x = X_m / R
    na_y = Y_m / R

    # -------------------------
    # 4️⃣ 归一化 k-vectors
    # -------------------------
    kx_normalized = (n_medium * na_x / lambda_m) * recon_pixel_size_m
    ky_normalized = (n_medium * na_y / lambda_m) * recon_pixel_size_m

    #print(f"Calculated normalized kx range: [{kx_normalized.min():.3f}, {kx_normalized.max():.3f}]")
    #print(f"Calculated normalized ky range: [{ky_normalized.min():.3f}, {ky_normalized.max():.3f}]")

    # -------------------------
    # 5️⃣ 根据 loaded_led_indices 进行筛选
    # -------------------------
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
        print(">>> 启用刚体 K-Vector 校准模式 (Global Shift & Rotation)")
        if wavelength is None or recon_pixel_size is None:
            raise ValueError("刚体模式需要提供 wavelength 和 recon_pixel_size")
        
        # 确保坐标在设备上
        led_physics_coords = led_physics_coords.to(device)
        
        # 初始化 4 个刚体参数: [dx, dy, dz, theta]
        # 初始设为 0，代表没有额外位移
        rigid_params = torch.zeros(4, device=device, requires_grad=True)
        
        # 初始计算一次 kx, ky 以便后续使用
        curr_kx, curr_ky = compute_k_from_rigid_body(led_physics_coords, rigid_params, wavelength, recon_pixel_size)
    
    else:
        # 回退到旧模式 (直接使用传入的 kx, ky)
        if kx_batch is None or ky_batch is None:
            raise ValueError("如果不使用刚体模式，必须提供 kx_batch 和 ky_batch")
        curr_kx = kx_batch.detach().clone()
        curr_ky = ky_batch.detach().clone()
        if learn_k_vectors: # 旧的单点独立优化模式
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
        # 注意：这里我们先初始化，requires_grad 在循环里根据 epoch 切换
        learned_tensors.append({'params': rigid_params})
        base_max_lrs.append(1e-3) # 这里设为 1e-3，OneCycle 会从约 4e-5 开始增加
    elif learn_k_vectors:
        # 旧模式：独立优化每个 LED 的 kx, ky
        learned_tensors.append({'params': curr_kx})
        base_max_lrs.append(0.1)
        learned_tensors.append({'params': curr_ky})
        base_max_lrs.append(0.1)

    # 初始化优化器
    optimizer = torch.optim.AdamW(learned_tensors)

    # 初始化调度器：使用动态生成的 base_max_lrs 列表
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=base_max_lrs, 
        total_steps=epochs, 
        pct_start=0.3
    )
    
    metrics = {'loss': [], 'lr': [], 'dx': [], 'dy': [], 'dz': [], 'theta': []}



    if vis_interval:
        os.makedirs("output", exist_ok=True)
        snapshot_indices = list(range(0, epochs, vis_interval))
        if (epochs - 1) not in snapshot_indices:
            snapshot_indices.append(epochs - 1)
        
        num_snapshots = len(snapshot_indices)
        
        # 【修正 1】：必须放在 if vis_interval 内部
        fig, axes = plt.subplots(num_snapshots, 5, figsize=(22, 4 * num_snapshots)) 
        plt.subplots_adjust(hspace=0.4, wspace=0.3) 
        
        # 【修正 2】：记录初始 K 向量用于对比
        with torch.no_grad():
            if use_rigid_body:
                # 刚体模式：参数为 0 时的位置即初始位置
                init_kx, init_ky = compute_k_from_rigid_body(
                    led_physics_coords, torch.zeros(4, device=device), wavelength, recon_pixel_size
                )
            else:
                init_kx, init_ky = curr_kx, curr_ky
                
            init_kx_np = init_kx.cpu().numpy()
            init_ky_np = init_ky.cpu().numpy()
        
        snapshot_count = 0
    
    # 修改：4 列变为 5 列，figsize 宽度从 16 增加到 20
    fig, axes = plt.subplots(num_snapshots, 5, figsize=(20, 4 * num_snapshots)) 
    plt.subplots_adjust(hspace=0.3, wspace=0.3) 
    
    # 提前记录初始 K 向量作为对比基准
    with torch.no_grad():
        if use_rigid_body:
            init_kx, init_ky = compute_k_from_rigid_body(
                led_physics_coords, torch.zeros(4, device=device), wavelength, recon_pixel_size
            )
        else:
            init_kx, init_ky = kx_batch, ky_batch
            
        init_kx_np = init_kx.cpu().numpy()
        init_ky_np = init_ky.cpu().numpy()

    
    # Training loop
    loop = tqdm(range(epochs), desc="Solving")
    for _ in loop:
        # >>> 关键修改点：在 Forward 之前重新计算 K <<<
        if use_rigid_body:
            if _ > 100: 
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

        if use_rigid_body:
            # 物理约束：惩罚偏离原点。系数 1e5 是因为 dx^2 数值非常小(如 1e-6)
            # 这个正则项会拉住坐标，不让它乱飞
            # 将物理量除以它们的典型偏差范围，再平方
            scale_d = 1e-3  # 毫米级别的容忍度
            scale_theta = 0.05 # 约 3 度的容忍度
            reg_loss = ( (rigid_params[0]/scale_d)**2 + 
                        (rigid_params[1]/scale_d)**2 + 
                        (rigid_params[2]/scale_d)**2 + 
                        (rigid_params[3]/scale_theta)**2 ) * 0.1 # 控制整体权重
            #reg_loss = torch.sum(rigid_params**2) * 1e5 
            loss = loss + reg_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['loss'].append(loss.item())
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

                    # --- 第 5 列：LED 坐标 (K-Space) 可视化 ---
                    ax_k = axes[snapshot_count, 4]
                    
                    # 绘制初始位置 (灰色圆点，背景参考)
                    ax_k.scatter(init_kx_np, init_ky_np, c='gray', s=12, alpha=0.3, label='Initial')
                    
                    # 绘制当前位置 (红色 X，实时变化)
                    curr_kx_v = curr_kx.detach().cpu().numpy()
                    curr_ky_v = curr_ky.detach().cpu().numpy()
                    ax_k.scatter(curr_kx_v, curr_ky_v, c='red', s=18, marker='x', linewidths=1, label='Current')

                    # 自动设置坐标轴范围，稍微留白
                    limit = max(np.abs(init_kx_np).max(), np.abs(init_ky_np).max()) * 1.3
                    ax_k.set_xlim(-limit, limit)
                    ax_k.set_ylim(-limit, limit)
                    ax_k.set_aspect('equal')
                    ax_k.grid(True, linestyle=':', alpha=0.5)
                    ax_k.set_title(f"Ep {_} | LED Geometry")

                    # 如果是刚体变换，在 X 轴标签显示物理位移量
                    if use_rigid_body:
                        p = rigid_params.detach().cpu().numpy()
                        # dx, dy 换算为 mm 显示，theta 换算为角度
                        info_str = f"dx:{p[0]*1e3:.2f}mm dy:{p[1]*1e3:.2f}mm\ndz:{p[2]*1e3:.2f}mm rot:{np.degrees(p[3]):.2f}°"
                        ax_k.set_xlabel(info_str, fontsize=9, color='blue')

                    # 只有第一行显示图例，避免重复
                    if snapshot_count == 0:
                        ax_k.legend(loc='upper right', fontsize=8)

                    # 统一处理坐标轴开关
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

    if use_rigid_body:
        with torch.no_grad():
             final_kx, final_ky = compute_k_from_rigid_body(led_physics_coords, rigid_params, wavelength, recon_pixel_size)
    else:
        final_kx, final_ky = curr_kx.detach(), curr_ky.detach()

    return object.detach(), pupil.detach(), final_kx, final_ky, metrics

def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    """计算 3D 复合张量的全变分 (Total Variation)"""
    # 空间维度 TV (X 和 Y 方向)
    tv_h = torch.sum(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
    tv_w = torch.sum(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    # 深度维度 TV (Z 方向) - 使得相邻切片之间过渡平滑
    tv_d = torch.sum(torch.abs(x[1:, :, :] - x[:-1, :, :])) if x.shape[0] > 1 else 0.0
    return tv_h + tv_w + tv_d

def solve_inverse_slice(
    captures: Float[torch.Tensor, "B n n"],
    object: Complex[torch.Tensor, "D N N"], # >>> 3D 修改：支持 D 层切片 <<<
    pupil: Complex[torch.Tensor, "N N"],
    led_physics_coords: Float[torch.Tensor, "B 3"] | None = None, 
    wavelength: float | None = None,       
    recon_pixel_size: float | None = None, 
    slice_spacing: float = 0.0,            # >>> 3D 修改：切片物理间距 (米) <<<
    kx_batch: Float[torch.Tensor, "B"] | None = None, 
    ky_batch: Float[torch.Tensor, "B"] | None = None,
    
    learn_pupil: bool = True,
    learn_k_vectors: bool = False, 
    epochs: int = 500,
    vis_interval: int = 0,
    tv_weight: float = 1e-4,               # >>> 3D 修改：TV正则化权重 <<<
) -> tuple[Complex[torch.Tensor, "D N N"], Complex[torch.Tensor, "N N"], Float[torch.Tensor, "B"], Float[torch.Tensor, "B"], dict[str, list[float]]]:

    # 提取切片数和输出尺寸
    D, output_size, _ = object.shape
    downsample_factor = output_size // captures[0].shape[0]
    device = object.device
    
    # --- 1. 参数校验与初始化 ---
    use_rigid_body = learn_k_vectors and (led_physics_coords is not None)
    
    if use_rigid_body:
        print(">>> 启用刚体 K-Vector 校准模式 (Global Shift & Rotation)")
        if wavelength is None or recon_pixel_size is None:
            raise ValueError("刚体模式需要提供 wavelength 和 recon_pixel_size")
        
        led_physics_coords = led_physics_coords.to(device)
        rigid_params = torch.zeros(4, device=device, requires_grad=True)
        curr_kx, curr_ky = compute_k_from_rigid_body(led_physics_coords, rigid_params, wavelength, recon_pixel_size)
    else:
        if kx_batch is None or ky_batch is None:
            raise ValueError("如果不使用刚体模式，必须提供 kx_batch 和 ky_batch")
        curr_kx = kx_batch.detach().clone()
        curr_ky = ky_batch.detach().clone()
        if learn_k_vectors: 
             curr_kx.requires_grad_(True)
             curr_ky.requires_grad_(True)
             
        # 安全检查：如果在 3D 模式下未提供波长和像素尺寸，将无法计算 ASM 传播
        if D > 1 and (wavelength is None or recon_pixel_size is None):
            raise ValueError("3D Multi-slice 模式必须提供 wavelength 和 recon_pixel_size 以计算空间传播")

    # --- 2. 优化器设置 ---
    learned_tensors = []
    base_max_lrs = [] 

    # 1. Object 参数
    object = object.clone().detach().requires_grad_(True)
    learned_tensors.append({'params': object})
    base_max_lrs.append(0.1) 

    # 2. Pupil 参数 
    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': pupil})
        base_max_lrs.append(0.1)

    # 3. 刚体变换或 K 向量参数 
    if use_rigid_body:
        learned_tensors.append({'params': rigid_params})
        base_max_lrs.append(1e-3) 
    elif learn_k_vectors:
        learned_tensors.append({'params': curr_kx})
        base_max_lrs.append(0.1)
        learned_tensors.append({'params': curr_ky})
        base_max_lrs.append(0.1)

    optimizer = torch.optim.AdamW(learned_tensors)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_max_lrs, total_steps=epochs, pct_start=0.3
    )
    
    metrics = {'loss': [], 'lr': [], 'dx': [], 'dy': [], 'dz': [], 'theta': []}

    if vis_interval:
        os.makedirs("output", exist_ok=True)
        snapshot_indices = list(range(0, epochs, vis_interval))
        if (epochs - 1) not in snapshot_indices:
            snapshot_indices.append(epochs - 1)
        
        num_snapshots = len(snapshot_indices)
        snapshot_count = 0
    
        fig, axes = plt.subplots(num_snapshots, 5, figsize=(20, 4 * num_snapshots)) 
        if num_snapshots == 1:
            axes = np.expand_dims(axes, 0) # 处理单行 subplots 的维度问题
        plt.subplots_adjust(hspace=0.3, wspace=0.3) 
    
        with torch.no_grad():
            if use_rigid_body:
                init_kx, init_ky = compute_k_from_rigid_body(
                    led_physics_coords, torch.zeros(4, device=device), wavelength, recon_pixel_size
                )
            else:
                init_kx, init_ky = curr_kx, curr_ky
                
            init_kx_np = init_kx.cpu().numpy()
            init_ky_np = init_ky.cpu().numpy()

    # --- 3. Training loop ---
    loop = tqdm(range(epochs), desc=f"Solving ({D} Slices)")
    for _ in loop:
        if use_rigid_body:
            if _ > 100: 
                rigid_params.requires_grad_(True)
                curr_kx, curr_ky = compute_k_from_rigid_body(
                    led_physics_coords, rigid_params, wavelength, recon_pixel_size
                )
            else:
                rigid_params.requires_grad_(False)
                curr_kx, curr_ky = compute_k_from_rigid_body(
                    led_physics_coords, torch.zeros_like(rigid_params), wavelength, recon_pixel_size
                )
            metrics['dx'].append(rigid_params[0].item())
            metrics['dy'].append(rigid_params[1].item())
            metrics['dz'].append(rigid_params[2].item())
            metrics['theta'].append(rigid_params[3].item())

        # >>> 3D 修改：调用多层传播前向模型 <<<
        predicted_intensities = forward_model_multislice(
            object_tensor=object, 
            pupil_tensor=pupil, 
            kx=curr_kx, 
            ky=curr_ky, 
            wavelength=wavelength,
            recon_pixel_size=recon_pixel_size,
            slice_spacing=slice_spacing,
            downsample_factor=downsample_factor
        )

        # 数据拟合 Loss
        data_loss = torch.nn.functional.l1_loss(predicted_intensities, captures)
        
        # >>> 3D 修改：加入水凝胶背景 TV 约束 <<<
        reg_tv_loss = tv_weight * compute_tv_loss(object)
        loss = data_loss + reg_tv_loss

        if use_rigid_body:
            scale_d = 1e-3 
            scale_theta = 0.05 
            reg_rigid_loss = ( (rigid_params[0]/scale_d)**2 + 
                        (rigid_params[1]/scale_d)**2 + 
                        (rigid_params[2]/scale_d)**2 + 
                        (rigid_params[3]/scale_theta)**2 ) * 0.1 
            loss = loss + reg_rigid_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics['loss'].append(loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

        if vis_interval and _ in snapshot_indices:
            with torch.no_grad():
                # >>> 3D 修改：提取中心切片进行可视化 <<<
                mid_slice = D // 2
                obj_np = object[mid_slice].detach().cpu()
                obj_amp = torch.abs(obj_np).numpy()
                obj_phase = torch.angle(obj_np).numpy()

                pup_np = pupil.detach().cpu()
                pup_amp = torch.abs(pup_np).numpy()
                pup_phase = torch.angle(pup_np).numpy()

                axes[snapshot_count, 0].imshow(obj_amp, cmap='gray')
                axes[snapshot_count, 0].set_title(f"Ep {_} | Slice {mid_slice} Amp")
                
                axes[snapshot_count, 1].imshow(obj_phase, cmap='viridis')
                axes[snapshot_count, 1].set_title(f"Ep {_} | Slice {mid_slice} Phase")
                
                axes[snapshot_count, 2].imshow(pup_amp, cmap='gray')
                axes[snapshot_count, 2].set_title(f"Ep {_} | Pup Amp")
                
                axes[snapshot_count, 3].imshow(pup_phase, cmap='magma') 
                axes[snapshot_count, 3].set_title(f"Ep {_} | Pup Phase")

                ax_k = axes[snapshot_count, 4]
                ax_k.scatter(init_kx_np, init_ky_np, c='gray', s=12, alpha=0.3, label='Initial')
                
                curr_kx_v = curr_kx.detach().cpu().numpy()
                curr_ky_v = curr_ky.detach().cpu().numpy()
                ax_k.scatter(curr_kx_v, curr_ky_v, c='red', s=18, marker='x', linewidths=1, label='Current')

                limit = max(np.abs(init_kx_np).max(), np.abs(init_ky_np).max()) * 1.3
                ax_k.set_xlim(-limit, limit)
                ax_k.set_ylim(-limit, limit)
                ax_k.set_aspect('equal')
                ax_k.grid(True, linestyle=':', alpha=0.5)
                ax_k.set_title(f"Ep {_} | LED Geometry")

                if use_rigid_body:
                    p = rigid_params.detach().cpu().numpy()
                    info_str = f"dx:{p[0]*1e3:.2f}mm dy:{p[1]*1e3:.2f}mm\ndz:{p[2]*1e3:.2f}mm rot:{np.degrees(p[3]):.2f}°"
                    ax_k.set_xlabel(info_str, fontsize=9, color='blue')

                if snapshot_count == 0:
                    ax_k.legend(loc='upper right', fontsize=8)

                for col in range(4):
                    axes[snapshot_count, col].axis('off')
                
                snapshot_count += 1

    if vis_interval:
        save_path = "output/iteration_process_dense.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) 
        print(f"\nIteration progress saved to: {save_path}")

    final_object = object.detach()
    final_pupil = pupil.detach()

    if use_rigid_body:
        with torch.no_grad():
             final_kx, final_ky = compute_k_from_rigid_body(led_physics_coords, rigid_params, wavelength, recon_pixel_size)
    else:
        final_kx, final_ky = curr_kx.detach(), curr_ky.detach()

    return final_object, final_pupil, final_kx, final_ky, metrics


