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


def visualize_kspace_and_captures(
    captures: torch.Tensor,
    kx_normalized: torch.Tensor,
    ky_normalized: torch.Tensor,
    output_filename: str = "output/capture_orientation_validation.png"
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
    metrics_file_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # 画曲线
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=key)

    plt.title("Training Metrics")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "real_data_metrics_curve.png"))
    plt.close()