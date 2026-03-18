import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================
N = 256                 # 图像尺寸
dx = 1e-6               # 空间采样间隔
wavelength = 532e-9     # 波长
k0 = 2 * np.pi / wavelength

n_gel = 1.33
n_sphere = 1.59
delta_n = n_sphere - n_gel

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 坐标网格
# =========================
x = torch.linspace(-N//2, N//2-1, N) * dx
y = torch.linspace(-N//2, N//2-1, N) * dx
X, Y = torch.meshgrid(x, y, indexing='ij')
X = X.to(device)
Y = Y.to(device)

# =========================
# 生成微球分布
# =========================
def generate_spheres(num=5, radius=5e-6):
    obj = torch.zeros((N, N), device=device)
    for _ in range(num):
        cx = (torch.rand(1, device=device) - 0.5) * N * dx * 0.6
        cy = (torch.rand(1, device=device) - 0.5) * N * dx * 0.6
        sphere = ((X - cx)**2 + (Y - cy)**2) < radius**2
        obj += sphere.float()
    return obj * delta_n
# =========================
# 光瞳函数
# =========================
def pupil_function(NA=0.1):
    fx = torch.fft.fftfreq(N, dx).to(device)
    fy = torch.fft.fftfreq(N, dx).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    cutoff = NA / wavelength
    pupil = (FX**2 + FY**2) < cutoff**2
    return pupil.float()

# =========================
# 倾斜照明
# =========================
def tilted_illumination(theta_x, theta_y):
    kx = k0 * np.sin(theta_x)
    ky = k0 * np.sin(theta_y)
    return torch.exp(1j * (kx * X + ky * Y))

# =========================
# 成像模型（Born近似）
# =========================
def forward_model(obj, theta_x, theta_y, pupil):
    U_in = tilted_illumination(theta_x, theta_y)

    # 散射项
    scatter = obj * U_in

    F_scatter = torch.fft.fft2(scatter)
    U_out = U_in + torch.fft.ifft2(F_scatter)

    # 成像系统
    U_img = torch.fft.ifft2(torch.fft.fft2(U_out) * pupil)
    I = torch.abs(U_img) ** 2
    return I

# =========================
# 主程序
# =========================
obj = generate_spheres(num=8)
pupil = pupil_function(NA=0.1)

angles = [
    (0.0, 0.0),
    (0.05, 0.0),
    (-0.5, 0.0),
    (0.0, 0.05),
    (0.0, -0.5),
]

plt.figure(figsize=(10, 6))

for i, (tx, ty) in enumerate(angles):
    img = forward_model(obj, tx, ty, pupil).cpu().numpy()

    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"theta=({tx:.2f},{ty:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()