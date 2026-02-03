from ptych.forward import forward_model
import matplotlib.pyplot as plt
#from ptych.utils import check_range
import torch
from tqdm import tqdm
from jaxtyping import Float, Complex
import os

def solve_inverse(
    captures: Float[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
    object: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    pupil: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
    epochs: int = 500,
    vis_interval: int = 5,
) -> tuple[Complex[torch.Tensor, "N N"], Complex[torch.Tensor, "N N"], dict[str, list[float]]]:

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
    fig, axes = plt.subplots(num_snapshots, 2, figsize=(10, 4 * num_snapshots))
    plt.subplots_adjust(hspace=0.3) 
    
    snapshot_count = 0
    # Training loop
    for _ in tqdm(range(epochs), desc="Solving"):
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
        metrics['loss'].append(total_loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

        # --- 3. 记录图片到网格 ---
        if _ in snapshot_indices:
            with torch.no_grad():
                obj_np = object.detach().cpu()
                amp = torch.abs(obj_np).numpy()
                phase = torch.angle(obj_np).numpy()

                # 绘制到对应的行
                ax_amp = axes[snapshot_count, 0]
                ax_phase = axes[snapshot_count, 1]

                ax_amp.imshow(amp, cmap='gray')
                ax_amp.set_title(f"Epoch {_} | Amp", fontsize=10)
                ax_amp.axis('off')

                ax_phase.imshow(phase, cmap='viridis')
                ax_phase.set_title(f"Epoch {_} | Phase", fontsize=10)
                ax_phase.axis('off')

                snapshot_count += 1

    # --- 4. 保存最终的高密度大图 ---
    save_path = "output/iteration_process_dense.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # 释放内存，不显示窗口
    
    print(f"\nIteration progress saved to: {save_path}")

    return object.detach(), pupil.detach(), metrics
