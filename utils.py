# utils.py

import glob
from PIL import Image
import numpy as np
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def get_default_device() -> torch.device:
    """获取默认的计算设备 (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_real_captures(folder_path: str, file_pattern: str = "*.tif") -> tuple[torch.Tensor, list[int]]:
    """
    从文件夹加载一系列图像，并根据文件名中的二进制编码解析LED索引。

    文件名示例: snapshot_..._1000...000.tif, 其中 '1' 的位置代表LED索引。

    Args:
        folder_path (str): 存放图像的文件夹路径。
        file_pattern (str): 匹配图像文件的模式，例如 "*.tif"。

    Returns:
        tuple[torch.Tensor, list[int]]: 
            - captures (torch.Tensor): 按LED索引排序的图像张量 [B, H, W]。
            - led_indices (list[int]): 与captures张量对应的LED索引列表 (从1开始)。
    """
    filepaths = glob.glob(f"{folder_path}/{file_pattern}")
    if not filepaths:
        raise FileNotFoundError(f"No images found in {folder_path} with pattern {file_pattern}")

    capture_data = []
    
    # 正则表达式，用于从文件名中提取最后的二进制串
    # 它会匹配一个下划线后跟着一串0和1，直到.tif结尾
    pattern = re.compile(r'_([01]+)\.tif$')

    for fpath in filepaths:
        match = pattern.search(fpath)
        if not match:
            print(f"Warning: Could not parse LED index from filename: {fpath}")
            continue

        binary_str = match.group(1)
        
        # 查找 '1' 的位置。find() 返回第一个匹配项的索引 (从0开始)
        # 我们+1使其与你的LED索引(从1开始)匹配
        try:
            # 确保字符串中只有一个 '1'
            if binary_str.count('1') != 1:
                print(f"Warning: Filename has multiple or zero '1's, skipping: {fpath}")
                continue
            
            led_index = binary_str.find('1') + 1
            
            with Image.open(fpath) as img:
                img_np = np.array(img.convert('L'), dtype=np.float32)
                capture_data.append({
                    'index': led_index,
                    'tensor': torch.from_numpy(img_np)
                })
        except Exception as e:
            print(f"Error processing file {fpath}: {e}")

    if not capture_data:
        raise ValueError("No valid captures could be loaded. Check filenames and patterns.")

    # 根据解析出的LED索引对数据进行排序
    capture_data.sort(key=lambda x: x['index'])

    # 将排序后的数据分离成张量和索引列表
    led_indices = [item['index'] for item in capture_data]
    captures_list = [item['tensor'] for item in capture_data]
    
    captures = torch.stack(captures_list, dim=0)

    print(f"Successfully loaded and sorted {len(captures)} images.")
    print(f"Detected LED indices: {led_indices[:5]}...{led_indices[-5:]}")
    
    return captures, led_indices

def calculate_k_vectors_from_positions(
    filepath: str,
    lambda_nm: float,
    magnification: float,
    camera_pixel_size_um: float,
    recon_pixel_size_m: float,
    center_led_index: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从LED物理位置CSV文件计算归一化的k-vectors。"""
    lambda_m = lambda_nm * 1e-9
    
    df = pd.read_csv(filepath)
    X_m = df['X'].values * 1e-3
    Y_m = df['Y'].values * 1e-3
    Z_m = df['Z'].values * 1e-3

    # --- 中心校准 ---
    center_idx_in_df = center_led_index - 1 # DataFrame 索引从0开始
    x_center = X_m[center_idx_in_df]
    y_center = Y_m[center_idx_in_df]
    print(f"Centering k-vectors around LED #{center_led_index} at (X={x_center*1e3:.2f}, Y={y_center*1e3:.2f}) mm")
    X_m_centered = X_m - x_center
    Y_m_centered = Y_m - y_center

    # --- 计算 NA ---
    # 使用独立计算，这在大多数情况下足够准确
    na_x = np.sin(np.arctan(X_m_centered / Z_m))
    na_y = np.sin(np.arctan(Y_m_centered / Z_m))

    # --- 转换为归一化 k-vectors ---
    # k_normalized 的单位是 "cycles per pixel"
    # 它代表了由该NA在样品平面上，每个像素经历的相位周期数
    kx_normalized = na_x / lambda_m * recon_pixel_size_m
    ky_normalized = na_y / lambda_m * recon_pixel_size_m

    print(f"Calculated normalized kx range: [{kx_normalized.min():.3f}, {kx_normalized.max():.3f}]")
    print(f"Calculated normalized ky range: [{ky_normalized.min():.3f}, {ky_normalized.max():.3f}]")

    return torch.from_numpy(kx_normalized).float(), torch.from_numpy(ky_normalized).float()

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
    output_filename: str = "tmp/capture_orientation_validation.png",
    arrow_scale: float = 500.0 
):
    """
    可视化每个捕获的原始图像，并在其上用箭头指示计算出的k-vector方向。

    Args:
        captures (torch.Tensor): 低分辨率图像张量 [B, H, W]。
        kx_normalized (torch.Tensor): 归一化的kx向量 [B]。
        ky_normalized (torch.Tensor): 归一化的ky向量 [B]。
        output_filename (str): 保存图像的文件名。
        arrow_scale (float): 控制箭头长度的缩放因子，需要根据图像大小调整。
    """
    print("Generating capture orientation validation plot...")
    
    num_captures = len(captures)
    grid_size = int(np.ceil(np.sqrt(num_captures)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    
    # 图像尺寸
    h, w = captures.shape[-2:]
    center_x, center_y = w / 2, h / 2

    # 准备数据以便绘图
    captures_np = captures.cpu().numpy()

    for i, ax in enumerate(axes.flat):
        if i < num_captures:
            # 显示原始捕获图像
            ax.imshow(captures_np[i], cmap='gray')

            # 获取当前k-vector
            kx = kx_normalized[i].item()
            ky = ky_normalized[i].item()

            # 计算箭头向量。k-vector代表光的传播方向，
            # 这里我们画一个从中心指向外的箭头来表示它。
            # 乘以 arrow_scale 来控制箭头的视觉长度。
            # 注意：图像坐标系中，y轴通常是向下的，所以ky可能需要反转
            arrow_dx = kx * arrow_scale
            arrow_dy = ky * arrow_scale # 如果y轴方向相反，这里用 -ky

            # 在图像中心绘制一个箭头
            # ax.arrow(x_start, y_start, dx, dy, ...)
            ax.arrow(
                center_x, 
                center_y, 
                arrow_dx, 
                arrow_dy,
                head_width=max(1, 0.05 * arrow_scale * np.sqrt(kx**2 + ky**2)), # 箭头头部大小
                head_length=max(1.5, 0.08 * arrow_scale * np.sqrt(kx**2 + ky**2)),
                fc='red', # 箭头填充颜色
                ec='red', # 箭头边框颜色
                linewidth=max(0.5, 0.01 * arrow_scale * np.sqrt(kx**2 + ky**2))
            )
            
            ax.set_title(f"Idx {i+1}", fontsize=8)

        ax.axis('off')

    plt.suptitle("Capture Orientation vs. Calculated k-vector", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename)
    print(f"Validation plot saved to {output_filename}")