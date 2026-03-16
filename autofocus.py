import torch
import torch.fft as fft

def estimate_defocus_from_phase_correlation(
    img_center: torch.Tensor, 
    img_tilted: torch.Tensor, 
    na_x: float, 
    na_y: float, 
    pixel_size_obj_m: float
) -> float:
    """
    利用相位相关法计算两张图像的相对平移，并推导离焦量 z。
    
    参数:
        img_center: 中心照明图像 (2D Tensor)
        img_tilted: 倾斜照明图像 (2D Tensor，建议选择 Brightfield 内的相邻 LED)
        na_x, na_y: 倾斜照明对应的 x 和 y 方向的 NA (数值孔径)
        pixel_size_obj_m: 物方面积下的等效像素尺寸 (米)
    返回:
        z: 估计的离焦距离 (米)
    """
    # 1. 计算 FFT
    F1 = fft.fft2(img_center)
    F2 = fft.fft2(img_tilted)
    
    # 2. 计算归一化互功率谱 (提纯相位)
    R = F1 * F2.conj()
    R = R / (torch.abs(R) + 1e-8)
    
    # 3. IFFT 回到空域寻找狄拉克峰
    r = fft.ifft2(R).real
    r = fft.fftshift(r) # 将零频移到中心
    
    # 4. 寻找峰值坐标 (整像素精度)
    H, W = r.shape
    max_idx = torch.argmax(r)
    y_peak = (max_idx // W).item()
    x_peak = (max_idx % W).item()
    
    # 计算相对于中心的像素偏移量
    dy_pix = y_peak - H // 2
    dx_pix = x_peak - W // 2
    
    # 转换为物方平移物理距离 (米)
    dx_m = dx_pix * pixel_size_obj_m
    dy_m = dy_pix * pixel_size_obj_m
    
    # 5. 根据平移量和照明角度计算离焦 z
    # 物理公式: dx = z * tan(theta_x)
    # 因为 NA = sin(theta)，所以 tan(theta) = NA / sqrt(1 - NA^2)
    tan_theta_x = na_x / (1 - na_x**2 - na_y**2 + 1e-8)**0.5
    tan_theta_y = na_y / (1 - na_x**2 - na_y**2 + 1e-8)**0.5
    
    # 分别从 x 和 y 方向估计 z，取加权平均或直接用绝对值较大的方向避免除零
    z_x = dx_m / tan_theta_x if abs(tan_theta_x) > 1e-3 else 0.0
    z_y = dy_m / tan_theta_y if abs(tan_theta_y) > 1e-3 else 0.0
    
    if abs(tan_theta_x) > abs(tan_theta_y):
        z_est = z_x
    else:
        z_est = z_y
        
    return z_est

def digital_refocus(
    complex_obj: torch.Tensor, 
    z_m: float, 
    wavelength_m: float, 
    pixel_size_m: float, 
    use_exact: bool = True
) -> torch.Tensor:
    """
    利用 Fresnel 传播将复振幅拉回焦平面。
    
    参数:
        complex_obj: 离焦的复振幅张量 [N, N]
        z_m: 需要补偿的离焦距离 (米)。如果是正数代表向前传播，如果要把离焦拉回来通常输入 -z。
        wavelength_m: 波长 (米)
        pixel_size_m: 重建复振幅的像素大小 (米)
        use_exact: 是否使用精确的非傍轴传播核
    """
    N = complex_obj.shape[0]
    device = complex_obj.device
    
    # 创建频率坐标网格
    fx = fft.fftfreq(N, d=pixel_size_m, device=device)
    fy = fft.fftfreq(N, d=pixel_size_m, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    if use_exact:
        # 精确版 (角谱法传递函数 Exact Angular Spectrum Kernel)
        # H(kx, ky) = exp(i * 2pi * z * sqrt((1/lambda)^2 - fx^2 - fy^2))
        term = (1.0 / wavelength_m)**2 - FX**2 - FY**2
        term = torch.clamp(term, min=0.0) # 滤除倏逝波 (Evanescent waves) 防止发散
        H = torch.exp(1j * 2 * torch.pi * z_m * torch.sqrt(term))
    else:
        # 抛物近似版 (Paraxial / Fresnel approximation)
        # H(kx, ky) = exp(-i * pi * lambda * z * (fx^2 + fy^2))
        H = torch.exp(-1j * torch.pi * wavelength_m * z_m * (FX**2 + FY**2))
        
    # 频域相乘，空域重构
    O_f = fft.fft2(complex_obj)
    O_refocused = fft.ifft2(O_f * H)
    
    return O_refocused