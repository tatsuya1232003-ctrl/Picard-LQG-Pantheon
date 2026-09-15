import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad
import urllib.request
import matplotlib.pyplot as plt

# =====================================================================
# 1. TẢI VÀ XỬ LÝ DỮ LIỆU PANTHEON+ FULL COVARIANCE
# =====================================================================
print("Đang tải dữ liệu Pantheon+ và ma trận covariance...")
url_data = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
url_cov = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

df = pd.read_csv(url_data, sep=r'\s+')
response = urllib.request.urlopen(url_cov)
lines = response.readlines()
N_points = int(lines[0].decode('utf-8').strip())
cov_1d = np.loadtxt(lines[1:])
C_matrix = cov_1d.reshape((N_points, N_points))

df['original_index'] = np.arange(len(df))
data_filtered = df[df['zHD'] > 0.01].copy().sort_values(by='zHD')
keep_indices = data_filtered['original_index'].values

C_filtered = C_matrix[keep_indices, :][:, keep_indices]
C_inv = np.linalg.inv(C_filtered)

z_obs = data_filtered['zHD'].values
mu_obs = data_filtered['MU_SH0ES'].values
N = len(z_obs)
c = 299792.458  # km/s

print(f"-> Đã xử lý thành công: {N} Supernovae.\n")

# =====================================================================
# 2. MÔ HÌNH PICARD HORN VỚI PLANCKIAN CUSP CUT-OFF
# =====================================================================
DZ_FIXED = 0.0100  # Minimum percolation scale

def f_quantum_cutoff(z, alpha, z_c, z_cut):
    """Metric modification f(z) tích hợp Sharp Transition & Planck Cusp Cut-off"""
    sigmoid = 1.0 / (1.0 + np.exp(-(z - z_c) / DZ_FIXED))
    cutoff = np.exp(-z / z_cut)
    return (1.0 + alpha * sigmoid) * cutoff

def d_comoving_cutoff_single(z_target, alpha, z_c, z_cut):
    if z_target <= 0:
        return 0.0
    integrand = lambda zp: f_quantum_cutoff(zp, alpha, z_c, z_cut) / (1.0 + zp)
    val, _ = quad(integrand, 0, z_target)
    return val

def dL_picard_cutoff(z_array, H0, alpha, z_c, z_cut):
    dL_list = []
    for zi in z_array:
        d_com = d_comoving_cutoff_single(zi, alpha, z_c, z_cut)
        dL_val = (c / H0) * (1.0 + zi) * np.sinh(d_com)
        dL_list.append(dL_val)
    return np.array(dL_list)

def mu_picard_cutoff(z_array, H0, alpha, z_c, z_cut):
    dL = dL_picard_cutoff(z_array, H0, alpha, z_c, z_cut)
    return 5.0 * np.log10(dL) + 25.0

def chi2_picard_cutoff(params):
    H0, alpha, z_c, z_cut = params
    delta = mu_obs - mu_picard_cutoff(z_obs, H0, alpha, z_c, z_cut)
    return delta.T @ C_inv @ delta

# =====================================================================
# 3. LAMBDA-CDM BASELINE
# =====================================================================
def dL_lcdm_single(z_single, H0, Om):
    integrand = lambda zp: 1.0 / np.sqrt(Om * (1.0 + zp)**3 + (1.0 - Om))
    integral, _ = quad(integrand, 0, z_single)
    return (c / H0) * (1.0 + z_single) * integral

def mu_lcdm(z_array, H0, Om):
    dL_vals = np.array([dL_lcdm_single(zi, H0, Om) for zi in z_array])
    return 5.0 * np.log10(dL_vals) + 25.0

def chi2_lcdm(params):
    H0, Om = params
    delta = mu_obs - mu_lcdm(z_obs, H0, Om)
    return delta.T @ C_inv @ delta

# =====================================================================
# 4. TỐI ƯU HÓA HỆ THỐNG (L-BFGS-B)
# =====================================================================
print("Đang chạy tối ưu hóa Picard Metric có Cusp Cut-off (k = 4)...")

# Khởi tạo: H0~72.6, alpha~0.038, z_c~0.068, z_cut~12.5 (dựa trên Planck scale volume)
initial_params = [72.6, 0.038, 0.068, 12.5]
bounds_params = [(60.0, 80.0), (-0.2, 0.2), (0.01, 0.5), (3.0, 50.0)]

res_picard = minimize(chi2_picard_cutoff, initial_params, bounds=bounds_params, method='L-BFGS-B')

chi2_p = res_picard.fun
H0_p, alpha_p, zc_p, zcut_p = res_picard.x
k_p = 4
aic_p = chi2_p + 2 * k_p
bic_p = chi2_p + k_p * np.log(N)

print("Đang chạy tối ưu hóa Lambda-CDM...")
res_lcdm = minimize(chi2_lcdm, [73.0, 0.3], bounds=[(50.0, 90.0), (0.1, 0.5)])
chi2_l = res_lcdm.fun
H0_l, Om_l = res_lcdm.x
k_l = 2
aic_l = chi2_l + 2 * k_l
bic_l = chi2_l + k_l * np.log(N)

# =====================================================================
# 5. IN BẢNG KẾT QUẢ CẬP NHẬT
# =====================================================================
print("\n" + "="*75)
print(f"{'Thông số / Chỉ số':<30} | {'Picard Cut-off (k=4)':<20} | {'Lambda-CDM (k=2)':<18}")
print("="*75)
print(f"{'Số tham số tự do (k)':<30} | {k_p:<20} | {k_l:<18}")
print(f"{'H0 (km/s/Mpc)':<30} | {H0_p:<20.3f} | {H0_l:<18.3f}")
print(f"{'Biên độ chuyển pha (alpha)':<30} | {alpha_p:<20.4f} | {'N/A':<18}")
print(f"{'Redshift tới hạn (z_c)':<30} | {zc_p:<20.4f} | {'N/A':<18}")
print(f"{'Độ rộng chuyển pha (dz)':<30} | {DZ_FIXED:<20.4f} (Fixed) | {'N/A':<18}")
print(f"{'Planck Cusp Cut-off (z_cut)':<30} | {zcut_p:<20.3f} | {'N/A':<18}")
print(f"{'Tham số Omega_m':<30} | {'N/A':<20} | {Om_l:<18.4f}")
print("-" * 75)
print(f"{'Tổng Chi-squared (Chi2)':<30} | {chi2_p:<20.2f} | {chi2_l:<18.2f}")
print(f"{'Chi2 / dof':<30} | {chi2_p/(N-k_p):<20.3f} | {chi2_l/(N-k_l):<18.3f}")
print(f"{'Akaike Info Criterion (AIC)':<30} | {aic_p:<20.2f} | {aic_l:<18.2f}")
print(f"{'Bayesian Info Criterion (BIC)':<30} | {bic_p:<20.2f} | {bic_l:<18.2f}")
print("="*75)

# =====================================================================
# 6. VẼ ĐỒ THỊ MỚI (HIGH-Z RESIDUALS ĐÃ PHẲNG)
# =====================================================================
print("\nĐang tạo đồ thị đối chiếu Residuals đã khắc phục high-z...")

mu_model_picard = mu_picard_cutoff(z_obs, H0_p, alpha_p, zc_p, zcut_p)
mu_model_lcdm = mu_lcdm(z_obs, H0_l, Om_l)
res_picard = mu_obs - mu_model_picard

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.errorbar(z_obs, mu_obs, yerr=np.sqrt(np.diag(C_filtered)), fmt='.', color='gray', alpha=0.15, label=r'Pantheon+ SNe Ia Data')
ax1.plot(z_obs, mu_model_picard, 'r-', lw=2, label=r'Picard Metric with Cusp Cut-off ($H_0 = ' + f'{H0_p:.2f}' + r'$)')
ax1.plot(z_obs, mu_model_lcdm, 'b--', lw=2, label=r'$\Lambda$CDM Model ($H_0 = ' + f'{H0_l:.2f}' + r', \Omega_m = ' + f'{Om_l:.2f}' + r'$)')
ax1.set_ylabel(r'Distance Modulus $\mu$', fontsize=12)
ax1.set_title(r'Pantheon+ Fit: Picard Horn with Planckian Cusp Cut-off vs $\Lambda$CDM', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.scatter(z_obs, res_picard, color='red', s=8, alpha=0.3, label=r'Picard Residuals (Planck Cut-off)')
ax2.axhline(0, color='blue', linestyle='--', lw=1.5, label=r'$\Lambda$CDM Baseline ($\Delta\mu = 0$)')
ax2.set_xlabel(r'Redshift $z$', fontsize=12)
ax2.set_ylabel(r'$\Delta\mu$ (Data - Model)', fontsize=10)
ax2.set_ylim(-1.0, 1.0)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('picard_pantheon_fit_cutoff.pdf', dpi=300)
plt.show()
