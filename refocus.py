import torch
import torch.fft as fft
'''
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
'''

def estimate_defocus_from_phase_correlation(
    img_center: torch.Tensor, 
    img_tilted: torch.Tensor, 
    na_x: float, 
    na_y: float, 
    pixel_size_obj_m: float
) -> float:
    # === 关键修复 1：减去均值 (去除 DC 分量致盲效应) ===
    img1 = img_center - torch.mean(img_center)
    img2 = img_tilted - torch.mean(img_tilted)
    
    # === 关键修复 2：加二维汉明窗 (抑制傅里叶变换的边缘截断伪影) ===
    H, W = img1.shape
    win_y = torch.hann_window(H, device=img1.device).view(H, 1)
    win_x = torch.hann_window(W, device=img1.device).view(1, W)
    window = win_y * win_x
    
    img1 = img1 * window
    img2 = img2 * window

    # 1. 计算 FFT
    F1 = fft.fft2(img1)
    F2 = fft.fft2(img2)
    
    # 2. 计算归一化互功率谱 (提纯相位)
    R = F1 * F2.conj()
    R = R / (torch.abs(R) + 1e-8)
    
    # 3. IFFT 回到空域寻找狄拉克峰
    r = fft.ifft2(R).real
    r = fft.fftshift(r) # 将零频移到中心
    
    # 4. 寻找峰值坐标 (整像素精度)
    max_idx = torch.argmax(r)
    y_peak = (max_idx // W).item()
    x_peak = (max_idx % W).item()
    
    # === 亚像素插值 (Sub-pixel Interpolation) ===
    def quadratic_interp(val_m1, val_0, val_p1):
        denom = 2 * (val_m1 - 2 * val_0 + val_p1)
        if abs(denom) > 1e-6:
            return (val_m1 - val_p1) / denom
        return 0.0

    delta_x = quadratic_interp(r[y_peak, x_peak - 1].item(), r[y_peak, x_peak].item(), r[y_peak, x_peak + 1].item()) if 0 < x_peak < W - 1 else 0.0
    delta_y = quadratic_interp(r[y_peak - 1, x_peak].item(), r[y_peak, x_peak].item(), r[y_peak + 1, x_peak].item()) if 0 < y_peak < H - 1 else 0.0

    dx_pix = (x_peak + delta_x) - W // 2
    dy_pix = (y_peak + delta_y) - H // 2
    # ===================================================
    
    # 转换为物方平移物理距离 (米)
    dx_m = dx_pix * pixel_size_obj_m
    dy_m = dy_pix * pixel_size_obj_m
    
    # 5. 根据平移量和照明角度计算离焦 z
    tan_theta_x = na_x / (1 - na_x**2 - na_y**2 + 1e-8)**0.5
    tan_theta_y = na_y / (1 - na_x**2 - na_y**2 + 1e-8)**0.5
    
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