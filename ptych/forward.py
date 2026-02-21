# ptych\forward.py
"""
The FP forward model: O, P, {k_i}  →  {I_i}
"""
from typing import Callable, cast
from functools import partial
import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))
fftshift = cast(Callable[..., torch.Tensor], torch.fft.fftshift)
ifftshift = cast(Callable[..., torch.Tensor], torch.fft.ifftshift)

def forward_model(
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx: Float[torch.Tensor, "B"],
    ky: Float[torch.Tensor, "B"],
    downsample_factor: int = 1
) -> Complex[torch.Tensor, "B N/{downsample_factor} N/{downsample_factor}"]:
    """
    Forward model - returns images at each k-space location given an object

    Args:
        object_tensor (torch.Tensor): Object tensor [N, N] (0, 1)
        pupil_tensor (torch.Tensor): Pupil tensor [N, N]
        kx (torch.Tensor): Wavevector shift(s) in x direction, normalized. Tensor [B] (-0.5, 0.5)
        ky (torch.Tensor): Wavevector shift(s) in y direction, normalized. Tensor [B] (-0.5, 0.5)
        downsampling_factor (int): Downsampling factor for the output images

    Returns:
        torch.Tensor: Predicted intensities [B, N/downsample_factor, N/downsample_factor]
    """

    N, _ = object_tensor.shape
    dtype = object_tensor.dtype
    kx_reshaped = kx.view(-1, 1, 1)
    ky_reshaped = ky.view(-1, 1, 1)

    # Create coordinate grids [N, N]
    coords = torch.arange(N, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

    # Create phase ramps for all k-vectors at once
    # Phase ramp: exp(i * 2π * (kx*x + ky*y) / N)
    # Shape: [B, N, N]
    phase = 2 * torch.pi * (kx_reshaped * x_grid[None, :, :] + ky_reshaped * y_grid[None, :, :])
    phase_ramps = torch.exp(1j * phase.to(dtype))  # [B, N, N]


    # Apply phase ramps to object (multiply in spatial domain = shift in frequency domain)
    tilted_objects = object_tensor[None, :, :] * phase_ramps  # [B, N, N]

    # Batch FFT all tilted objects
    objects_fourier = fftshift(fft2(tilted_objects), dim=(-2, -1))  # [B, N, N]

    # Apply pupil filter (broadcast over batch dimension)
    filtered_fourier = pupil_tensor[None, :, :] * objects_fourier  # [B, N, N]

    # Batch inverse FFT
    #complex_image_fields = ifft2(filtered_fourier)  # [B, N, N]
    complex_image_fields = ifft2(ifftshift(filtered_fourier, dim=(-2, -1)))
    #gemini修改0221

    # Compute intensities
    predicted_intensities = torch.abs(complex_image_fields)**2  # [B, N, N]

    if downsample_factor > 1:
        predicted_intensities = F.avg_pool2d(predicted_intensities, kernel_size=downsample_factor, stride=downsample_factor)

    return predicted_intensities


# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))
fftshift = cast(Callable[..., torch.Tensor], torch.fft.fftshift)
ifftshift = cast(Callable[..., torch.Tensor], torch.fft.ifftshift)

def forward_model_multislice(
    object_tensor: Complex[torch.Tensor, "D N N"],  # 修改：D 表示切片层数
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx: Float[torch.Tensor, "B"],
    ky: Float[torch.Tensor, "B"],
    wavelength: float,          # 新增：光波长 (米)
    recon_pixel_size: float,    # 新增：重建像素物理尺寸 (米)
    slice_spacing: float,       # 新增：切片之间的物理间距 (米)
    downsample_factor: int = 1
) -> Float[torch.Tensor, "B N/{downsample_factor} N/{downsample_factor}"]: # 修正返回类型为 Float
    """
    3D Multi-Slice Forward model based on Angular Spectrum Method.
    """
    D, N, _ = object_tensor.shape
    device = object_tensor.device
    dtype = object_tensor.dtype

    # ==========================================
    # 1. 准备阶段：生成初始照明波 (Phase Ramps)
    # ==========================================
    kx_reshaped = kx.view(-1, 1, 1)
    ky_reshaped = ky.view(-1, 1, 1)

    coords = torch.arange(N, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

    # 生成倾斜平面波作为进入第 1 层的初始光场 [B, N, N]
    phase = 2 * torch.pi * (kx_reshaped * x_grid[None, :, :] + ky_reshaped * y_grid[None, :, :])
    field = torch.exp(1j * phase.to(dtype)) 

    # ==========================================
    # 2. 准备阶段：计算角谱传播核 H (ASM Kernel)
    # ==========================================
    if D > 1:
        # 使用 fftfreq 生成频率网格 (0, 1, ..., N/2, -N/2, ..., -1) / (N * pixel_size)
        # 这种排列天然匹配 fft2 输出，传播过程无需耗时的 fftshift
        fx = torch.fft.fftfreq(N, d=recon_pixel_size, device=device)
        fy = torch.fft.fftfreq(N, d=recon_pixel_size, device=device)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')

        f_sq = fx_grid**2 + fy_grid**2
        k_val = 1.0 / wavelength
        
        # 过滤掉倏逝波 (Evanescent waves)
        mask = (f_sq <= k_val**2).to(torch.float32)
        
        # 计算自由空间传递函数的相位延迟
        phase_H = 2 * torch.pi * slice_spacing * torch.sqrt(torch.clamp(k_val**2 - f_sq, min=0))
        H = mask * torch.exp(1j * phase_H.to(dtype))
        H = H.unsqueeze(0) # [1, N, N] 以便在 Batch 维度上广播

    # ==========================================
    # 3. 核心传播循环 (Multi-Slice Propagation)
    # ==========================================
    for d in range(D):
        # 步骤 A: 光波穿过当前切片 (空间域相乘)
        # field: [B, N, N], object_tensor[d]: [N, N] -> broadcast 
        field = field * object_tensor[d].unsqueeze(0)
        
        # 步骤 B: 传播到下一层 (如果是最后一层则跳过)
        if d < D - 1:
            field_f = fft2(field)    # 转换到频域
            field_f = field_f * H    # 乘以传递函数
            field = ifft2(field_f)   # 转换回空域，作为下一层的输入

    # ==========================================
    # 4. 出射波进入物镜 (Pupil Plane & Imaging)
    # ==========================================
    # 此时 field 是穿出样品的最终波前。将其移至频谱中心并应用光瞳
    exit_fourier = fftshift(fft2(field), dim=(-2, -1)) 
    filtered_fourier = pupil_tensor.unsqueeze(0) * exit_fourier 
    
    # 逆傅里叶变换回到相机平面 (注意这里的 ifftshift 修复了你之前的 bug)
    complex_image_fields = ifft2(ifftshift(filtered_fourier, dim=(-2, -1))) 

    # 计算强度
    predicted_intensities = torch.abs(complex_image_fields)**2 

    # 下采样 (如果需要加速，建议替换为之前讨论的频域裁剪截断法)
    if downsample_factor > 1:
        predicted_intensities = F.avg_pool2d(
            predicted_intensities, kernel_size=downsample_factor, stride=downsample_factor
        )

    return predicted_intensities