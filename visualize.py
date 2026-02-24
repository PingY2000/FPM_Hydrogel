# visualize.py

import glob
from PIL import Image
import numpy as np
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import os
import shutil
from pathlib import Path
import sys


def prepare_working_directories(output_dir="output", archive_dir="last_output"):
    """
    归档旧输出并重新初始化输出目录结构
    """
    output_path = Path(output_dir)
    archive_path = Path(archive_dir)

    # 1. 如果存在旧的 output，将其归档到 last_output
    if output_path.exists():
        # 如果已经存在旧的 last_output，先删除它以便更新
        if archive_path.exists():
            shutil.rmtree(archive_path)
        
        # 将当前的 output 整体重命名（移动）为 last_output
        output_path.rename(archive_path)
        print(f">>> 已将旧数据归档至: {archive_dir}")

    # 2. 创建全新的 output 及其子目录
    (output_path / "data").mkdir(parents=True, exist_ok=True)
    (output_path / "slice").mkdir(parents=True, exist_ok=True)
    
    print(f">>> 已初始化目录结构: {output_dir}/data 和 {output_dir}/slices")

def visualize_kspace_and_captures(
    captures: torch.Tensor,
    kx_normalized: torch.Tensor,
    ky_normalized: torch.Tensor,
    output_filename: str = "output/capture_orientation.png"
):
    print("Generating capture orientation validation plot...")
    
    num_captures = len(captures)
    grid_size = int(np.ceil(np.sqrt(num_captures)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    
    # 自动计算缩放因子：
    # 假设 k 向量的最大模长约为 0.5，我们希望最长的箭头占据图像半径的 80%
    h, w = captures.shape[-2:]
    center_x, center_y = w / 2, h / 2
    max_arrow_len = min(h, w) * 0.4  # 图像短边的一半再乘以 0.8
    
    # 获取所有 k 向量的模长，用于归一化长度比例
    k_norms = torch.sqrt(kx_normalized**2 + ky_normalized**2)
    max_k = torch.max(k_norms).item() if torch.max(k_norms) > 0 else 1.0
    # 这里的 scale 保证了物理意义上的比例：k 越大，箭头越长
    auto_scale = max_arrow_len / max_k 

    captures_np = captures.cpu().numpy()

    for i, ax in enumerate(axes.flat):
        if i < num_captures:
            ax.imshow(captures_np[i], cmap='gray')

            kx = kx_normalized[i].item()
            ky = ky_normalized[i].item()
            kn = k_norms[i].item()

            if kn < 1e-5:
                # 绘制中心点标记 (零向量照明)
                circle = Circle((center_x, center_y), radius=w*0.05, fill=False, color='cyan', linewidth=1.5)
                ax.add_patch(circle)
                ax.plot([center_x - w*0.03, center_x + w*0.03], [center_y, center_y], color='cyan', linewidth=1)
                ax.plot([center_x, center_x], [center_y - h*0.03, center_y + h*0.03], color='cyan', linewidth=1)
            else:
                # 计算位移向量：指向中心
                dx = kx * auto_scale
                dy = ky * auto_scale 

                # 逻辑说明：
                # 箭头的终点固定在中心 (center_x, center_y)
                # 起点则根据 k 向量的反方向偏移得出
                ax.annotate(
                    '', 
                    xy=(center_x, center_y),           # 箭头尖端位置 (中心)
                    xytext=(center_x - dx, center_y - dy), # 箭头尾部位置
                    arrowprops=dict(
                        arrowstyle='->, head_width=0.3, head_length=0.5',
                        color='red',
                        lw=1.5,
                        shrinkA=0, # 不在起点缩进
                        shrinkB=0  # 不在终点缩进
                    ),
                    zorder=10
                )
                
            ax.set_title(f"Idx {i}\n$k:[{kx:.2f}, {ky:.2f}]$", fontsize=7)
        ax.axis('off')

    plt.suptitle("Illumination Directions: Red arrows point TO center", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Validation plot saved to {output_filename}")
'''
def visualize_reconstruction(reconstructed_object, output_dir="output"):
    """
    可视化并保存最终重建的振幅和相位
    """
    os.makedirs(output_dir, exist_ok=True)

    final_amplitude = torch.abs(reconstructed_object)
    final_phase = torch.angle(reconstructed_object)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(final_amplitude.cpu().detach().numpy(), cmap='gray')
    axes[0].set_title("Reconstructed Amplitude")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(final_phase.cpu().detach().numpy(), cmap='viridis')
    axes[1].set_title("Reconstructed Phase")
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle("Final Reconstruction", fontsize=16)
    plt.savefig(os.path.join(output_dir, "final_reconstruction.png"))
    plt.close()

    # 单独保存无损图
    plt.imsave(
        os.path.join(output_dir, "final_amplitude_only.png"),
        final_amplitude.cpu().detach().numpy(),
        cmap='gray'
    )

    plt.imsave(
        os.path.join(output_dir, "final_phase_only.png"),
        final_phase.cpu().detach().numpy(),
        cmap='viridis'
    )
'''
def visualize_reconstruction(reconstructed_object, output_dir="output"):
    """
    可视化并保存最终重建结果。
    兼容 2D [N, N] 和 3D [D, N, N] 对象。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 统一转为 CPU numpy 方便处理
    obj_np = reconstructed_object.detach().cpu()
    
    # 如果是 2D，增加一个虚假的 D 维度，统一逻辑
    if obj_np.ndim == 2:
        obj_np = obj_np.unsqueeze(0)
    
    D, N, _ = obj_np.shape
    amplitudes = torch.abs(obj_np).numpy()
    phases = torch.angle(obj_np).numpy()

    # --- 1. 绘制平铺对比图 (所有切片) ---
    # 每行显示一个切片：[Amp, Phase]
    fig, axes = plt.subplots(D, 2, figsize=(12, 5 * D), squeeze=False)
    
    for d in range(D):
        im1 = axes[d, 0].imshow(amplitudes[d], cmap='gray')
        axes[d, 0].set_title(f"Slice {d} - Amplitude")
        fig.colorbar(im1, ax=axes[d, 0])

        im2 = axes[d, 1].imshow(phases[d], cmap='viridis')
        axes[d, 1].set_title(f"Slice {d} - Phase")
        fig.colorbar(im2, ax=axes[d, 1])

    plt.suptitle(f"Final 3D Reconstruction ({D} Slices)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "final_reconstruction.png"))
    plt.close()

    # --- 2. 绘制最大强度投影 (MIP) ---
    # 对于水凝胶中的细胞，MIP 可以一眼看到所有层中的物体
    if D > 1:
        mip_amp = np.max(amplitudes, axis=0)
        # 相位投影通常取平均或标准差，这里展示平均相位
        mean_phase = np.mean(phases, axis=0)

        fig_mip, axes_mip = plt.subplots(1, 2, figsize=(12, 6))
        axes_mip[0].imshow(mip_amp, cmap='gray')
        axes_mip[0].set_title("Maximum Intensity Projection (Amp)")
        axes_mip[1].imshow(mean_phase, cmap='viridis')
        axes_mip[1].set_title("Average Phase Projection")
        plt.savefig(os.path.join(output_dir, "final_mip_projection.png"))
        plt.close()

    # --- 3. 单独保存每一层为无损图 ---
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    for d in range(D):
        # 保存振幅
        plt.imsave(
            os.path.join(slices_dir, f"slice_{d}_amplitude.png"),
            amplitudes[d],
            cmap='gray'
        )
        # 保存相位
        plt.imsave(
            os.path.join(slices_dir, f"slice_{d}_phase.png"),
            phases[d],
            cmap='viridis'
        )

    if D > 1:
        print(f">>> 检测到多层切片，已生成 MIP 投影图并保存每一层至 /slices 目录")

def visualize_pupil(reconstructed_pupil, output_dir="output"):
    """
    可视化学习到的光瞳函数（幅度 + 相位）
    """
    os.makedirs(output_dir, exist_ok=True)

    learned_pupil_amp = torch.abs(reconstructed_pupil)
    learned_pupil_phase = torch.angle(reconstructed_pupil)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(learned_pupil_amp.cpu().detach().numpy(), cmap='gray')
    axes[0].set_title("Learned Pupil Amplitude")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(learned_pupil_phase.cpu().detach().numpy(), cmap='viridis')
    axes[1].set_title("Learned Pupil Phase (System Aberration)")
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle("Learned Pupil Function", fontsize=16)
    plt.savefig(os.path.join(output_dir, "learned_pupil.png"))
    plt.close()

def save_training_metrics(metrics: dict, output_dir: str = "output"):
    """
    保存并可视化训练过程中的指标曲线
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON
    metrics_file_path = os.path.join(output_dir, "data", "metrics.json")
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 画曲线
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=key)

    plt.title("Training Metrics")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "metrics_curve.png"))
    plt.close()

def plot_reconstruction_progress(snapshots, init_kx, init_ky, use_rigid_body=False, save_path="output/iteration_process.png"):
    """
    独立的可视化函数，用于生成迭代过程的高密度大图
    """
    if not snapshots:
        return

    #行显示状态
    save_msg = f"Saving final image to {save_path}..." 
    print(save_msg, end='', flush=True)

    num_snapshots = len(snapshots)
    # 5 列：Obj Amp, Obj Phase, Pup Amp, Pup Phase, LED Geometry
    fig, axes = plt.subplots(num_snapshots, 5, figsize=(20, 4 * num_snapshots))
    if num_snapshots == 1:
        axes = axes[np.newaxis, :] # 确保索引一致性
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    init_kx_np = init_kx.cpu().numpy()
    init_ky_np = init_ky.cpu().numpy()

    for i, snap in enumerate(snapshots):
        epoch = snap['epoch']
        
        # --- 1 & 2. Object 数据 ---
        obj_np = snap['object']
        axes[i, 0].imshow(np.abs(obj_np), cmap='gray')
        axes[i, 0].set_title(f"Ep {epoch} | Obj Amp")
        axes[i, 1].imshow(np.angle(obj_np), cmap='viridis')
        axes[i, 1].set_title(f"Ep {epoch} | Obj Phase")

        # --- 3 & 4. Pupil 数据 ---
        pup_np = snap['pupil']
        axes[i, 2].imshow(np.abs(pup_np), cmap='gray')
        axes[i, 2].set_title(f"Ep {epoch} | Pup Amp")
        axes[i, 3].imshow(np.angle(pup_np), cmap='magma')
        axes[i, 3].set_title(f"Ep {epoch} | Pup Phase")

        # --- 5. LED 坐标可视化 ---
        ax_k = axes[i, 4]
        ax_k.scatter(init_kx_np, init_ky_np, c='gray', s=12, alpha=0.3, label='Initial')
        
        curr_kx = snap['kx']
        curr_ky = snap['ky']
        ax_k.scatter(curr_kx, curr_ky, c='red', s=18, marker='x', linewidths=1, label='Current')

        limit = max(np.abs(init_kx_np).max(), np.abs(init_ky_np).max()) * 1.3
        ax_k.set_xlim(-limit, limit)
        ax_k.set_ylim(-limit, limit)
        ax_k.set_aspect('equal')
        ax_k.grid(True, linestyle=':', alpha=0.5)
        ax_k.set_title(f"Ep {epoch} | LED Geometry")

        if use_rigid_body and 'rigid_params' in snap:
            p = snap['rigid_params']
            info_str = f"dx:{p[0]*1e3:.2f}mm dy:{p[1]*1e3:.2f}mm\ndz:{p[2]*1e3:.2f}mm rot:{np.degrees(p[3]):.2f}°"
            ax_k.set_xlabel(info_str, fontsize=9, color='blue')

        if i == 0:
            ax_k.legend(loc='upper right', fontsize=8)

        # 关闭前四列的坐标轴
        for col in range(4):
            axes[i, col].axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    #删除行显示
    print("\r\033[K",end='', flush=True)