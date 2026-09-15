import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. TẬP DỮ LIỆU 32 COSMIC CHRONOMETERS (z, H(z), err_H)
cc_data = np.array([
    [0.070, 69.0, 19.6],  [0.090, 69.0, 12.0],  [0.120, 68.6, 26.2],
    [0.170, 83.0, 8.0],   [0.179, 75.0, 4.0],   [0.199, 75.0, 5.0],
    [0.200, 72.9, 29.6],  [0.240, 79.6, 2.6],   [0.270, 77.0, 14.0],
    [0.280, 88.8, 36.6],  [0.352, 83.0, 14.0],  [0.3802, 83.0, 13.5],
    [0.400, 95.0, 17.0],  [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2],
    [0.4497, 92.8, 12.9], [0.470, 89.0, 50.0],  [0.4783, 80.9, 9.0],
    [0.480, 97.0, 62.0],  [0.593, 104.0, 13.0], [0.680, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.880, 90.0, 40.0],
    [0.900, 117.0, 23.0], [1.037, 154.0, 20.0], [1.300, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.430, 177.0, 18.0], [1.530, 140.0, 14.0],
    [1.750, 202.0, 40.0], [1.965, 186.5, 50.4]
])

z_obs = cc_data[:, 0]
H_obs = cc_data[:, 1]
H_err = cc_data[:, 2]
N = len(z_obs)

# 2. KHAI BÁO HẰNG SỐ LÝ THUYẾT VÀ MÔ HÌNH
delta_z = 0.0100
z_cut = 5.80

def f_z(z, alpha, z_c):
    sigmoid = 1.0 / (1.0 + np.exp(-(z - z_c) / delta_z))
    return (1.0 + alpha * sigmoid) * np.exp(-z / z_cut)

def H_picard(z, H0, alpha, z_c):
    return H0 * (1.0 + z) / f_z(z, alpha, z_c)

def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))

def chi2_picard(params):
    H0, alpha, z_c = params
    H_th = H_picard(z_obs, H0, alpha, z_c)
    return np.sum(((H_obs - H_th) / H_err)**2)

def chi2_lcdm(params):
    H0, Om = params
    H_th = H_lcdm(z_obs, H0, Om)
    return np.sum(((H_obs - H_th) / H_err)**2)

# 3. TỐI ƯU HÓA TRÊN DỮ LIỆU CC
res_p = minimize(chi2_picard, [68.0, 0.09, 0.11], bounds=[(50.0, 90.0), (-0.5, 0.5), (0.01, 1.0)])
res_l = minimize(chi2_lcdm, [70.0, 0.3], bounds=[(50.0, 90.0), (0.1, 0.5)])

chi2_p = res_p.fun
H0_p, alpha_p, zc_p = res_p.x
k_p = 3

chi2_l = res_l.fun
H0_l, Om_l = res_l.x
k_l = 2

# 4. IN BẢNG KẾT QUẢ SO SÁNH
print("="*65)
print(f"{'Chỉ số (Metric)':<25} | {'Picard Horn (CC)':<18} | {'Lambda-CDM (CC)':<18}")
print("="*65)
print(f"{'H0 (km/s/Mpc)':<25} | {H0_p:<18.3f} | {H0_l:<18.3f}")
print(f"{'Tham số phụ (alpha / Om)':<25} | alpha = {alpha_p:<10.4f} | Om = {Om_l:<12.4f}")
print(f"{'z_c (Crit Redshift)':<25} | z_c = {zc_p:<12.4f} | N/A")
print(f"{'Chi-squared (Chi2)':<25} | {chi2_p:<18.2f} | {chi2_l:<18.2f}")
print(f"{'Chi2 / dof':<25} | {chi2_p/(N-k_p):<18.3f} | {chi2_l/(N-k_l):<18.3f}")
print("="*65)
print(f"Delta Chi2 (Picard - LCDM) trên 32 CC: {chi2_p - chi2_l:+.2f}")
print("="*65)

# 5. VẼ BIỂU ĐỒ H(z) VS REDSHIFT
z_grid = np.linspace(0.0, 2.1, 200)
plt.figure(figsize=(10, 6))
plt.errorbar(z_obs, H_obs, yerr=H_err, fmt='o', color='black', ecolor='gray', capsize=3, label='32 Cosmic Chronometers (Obs)')
plt.plot(z_grid, H_picard(z_grid, H0_p, alpha_p, zc_p), 'r-', linewidth=2, label=rf'Picard Horn ($H_0={H0_p:.2f}$)')
plt.plot(z_grid, H_lcdm(z_grid, H0_l, Om_l), 'b--', linewidth=2, label=rf'$\Lambda\text{{CDM}}$ ($H_0={H0_l:.2f}, \Omega_m={Om_l:.2f}$)')

plt.xlabel(r'Redshift $z$', fontsize=12)
plt.ylabel(r'$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
plt.title(r'Khớp mô hình $H(z)$ với 32 dữ liệu Cosmic Chronometers', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
