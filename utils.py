# utils.py

import glob
from PIL import Image
import numpy as np
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

def get_default_device() -> torch.device:
    """获取默认的计算设备 (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def decode_hex_to_led_index(hex_str: str) -> int:
    """
    核心解码逻辑：
    1. 十六进制转为整数
    2. 转为 43 位标准二进制字符串 (MSB first)
    3. 翻转字符串以还原“小端序” (LSB first)
    4. 查找 '1' 出现的位置并 +1 得到索引
    """
    val = int(hex_str, 16)
    # 填充为 43 位二进制
    bin_msb = f"{val:043b}"
    # 翻转，还原为小的位在前
    bin_lsb = bin_msb[::-1]
    
    index = bin_lsb.find('1') + 1
    return index

def load_real_captures(folder_path: str, file_pattern: str = "*.tif") -> tuple[torch.Tensor, list[int]]:
    """
    从文件夹加载图像，并解析曝光时间(exp)进行归一化，同时根据十六进制编码解析 LED 索引。
    文件名示例: 0210_162538_exp300_gain100_00000000001_G.tif
    """
    # 使用 os.path.join 提高兼容性
    search_path = os.path.join(folder_path, file_pattern)
    filepaths = glob.glob(search_path)
    
    if not filepaths:
        raise FileNotFoundError(f"No images found in {folder_path} with pattern {file_pattern}")

    capture_data = []
    
    # 修改后的正则表达式：
    # 匹配 exp(\d+), gain(\d+), 以及 11位十六进制 ([0-9a-fA-F]{11})
    pattern = re.compile(r'_exp(\d+)_gain(\d+)_([0-9a-fA-F]{11})_[A-Z]\.tif$')

    for fpath in filepaths:
        fname = os.path.basename(fpath)
        match = pattern.search(fname)
        if not match:
            # 如果不匹配新格式，尝试匹配旧格式以便兼容（可选）
            continue

        # 提取参数
        exp_val = float(match.group(1))   # 曝光时间
        gain_val = float(match.group(2)) # 增益 (通常FPM主要校准exp，gain若是非线性的则较难直接除)
        hex_str = match.group(3)
        
        try:
            # 1. 解码 LED 索引
            led_index = decode_hex_to_led_index(hex_str)
            if led_index <= 0: continue
            
            # 2. 读取图像
            with Image.open(fpath) as img:
                img_np = np.array(img.convert('L'), dtype=np.float32)
                
                # --- 核心曝光校准 ---
                # 将图像线性缩放到单位曝光时间下的亮度
                # 这样 exp300 的暗场和 exp30 的明场就能在同一量纲下对比
                if exp_val > 0:
                    img_np = img_np / exp_val
                else:
                    print(f"Warning: Zero exposure time in {fname}")
                # ------------------

                capture_data.append({
                    'index': led_index,
                    'tensor': torch.from_numpy(img_np)
                })
        except Exception as e:
            print(f"Error processing file {fpath}: {e}")

    if not capture_data:
        raise ValueError("No valid captures could be loaded. Check filenames (must contain 11-char hex).")

    # 根据解析出的 LED 索引对数据进行排序
    capture_data.sort(key=lambda x: x['index'])

    # 分离张量和索引
    led_indices = [item['index'] for item in capture_data]
    captures_list = [item['tensor'] for item in capture_data]
    
    captures = torch.stack(captures_list, dim=0)

    global_scale = captures.mean() 
    captures = captures / global_scale

    print(f"Successfully loaded and sorted {len(captures)} images.")
    print(f"Detected LED indices: {led_indices[:5]}...{led_indices[-5:]}")
    
    return captures, led_indices



def create_circular_pupil(shape: tuple[int, int], radius: int) -> torch.Tensor:
    """创建一个二元的圆形光瞳函数。"""
    N, M = shape
    coords_y, coords_x = torch.meshgrid(
        torch.arange(N, dtype=torch.float32) - N // 2,
        torch.arange(M, dtype=torch.float32) - M // 2,
        indexing='ij'
    )
    dist = torch.sqrt(coords_x**2 + coords_y**2)
    pupil = (dist < radius).to(torch.complex64)
    return pupil


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