#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_figures_fixed.py
宇宙图层叠加论（CLST）论文图片一键生成脚本
版本：1.1 (修复Q=0索引问题 & 加入校准分支)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import emcee
import corner
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 全局参数与物理常数（自然单位制，8πG=1, c=1）
# ============================================================
m_alpha = 0.6
rho_Lambda = 0.7 * 3.0
a_i = 1e-5
alpha_i = 0.0
alpha_dot_i = 0.0

def V(alpha):
    return 0.5 * m_alpha**2 * alpha**2
def Vp(alpha):
    return m_alpha**2 * alpha

# ============================================================
# 2. 核心微分方程组
# ============================================================
def system(t, y, Q):
    a, alpha, alpha_dot, rho_m = y
    H = Hubble(alpha, alpha_dot, rho_m, Q)
    if H <= 0:
        H = 1e-10
    da_dt = a * H
    dalpha_dt = alpha_dot
    dalpha_dot_dt = -3.0 * H * alpha_dot - Vp(alpha) - Q * rho_m
    denom = 1.0 - alpha * Q
    if abs(denom) < 1e-10:
        denom = 1e-10
    drho_m_dt = -3.0 * H * rho_m + (2.0 * Q * rho_m * alpha_dot) / denom
    return [da_dt, dalpha_dt, dalpha_dot_dt, drho_m_dt]

def Hubble(alpha, alpha_dot, rho_m, Q):
    rho_eff = (1.0 - alpha * Q) * rho_m + 0.5 * alpha_dot**2 + V(alpha) + rho_Lambda
    H2 = rho_eff / 3.0
    return np.sqrt(max(H2, 0.0))

# ============================================================
# 3. 积分至今天
# ============================================================
def integrate_to_present(rho_m_i, Q):
    t_span = (0.0, 200.0)
    y0 = [a_i, alpha_i, alpha_dot_i, rho_m_i]
    def hit_a1(t, y):
        return y[0] - 1.0
    hit_a1.terminal = True
    hit_a1.direction = 1

    sol = solve_ivp(lambda t, y: system(t, y, Q),
                    t_span, y0, method='DOP853',
                    events=hit_a1, rtol=1e-10, atol=1e-12,
                    dense_output=True)
    if not sol.success or len(sol.t_events[0]) == 0:
        return None

    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    t = sol.t
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(t))])
    H0 = H[-1]
    z = 1.0 / a - 1.0
    H_norm = H / H0

    Omega_m_eff = (1.0 - alpha[-1] * Q) * rho_m[-1] / (3.0 * H0**2)
    alpha0 = alpha[-1]
    alpha_dot0 = alpha_dot[-1]

    Hz_interp = interp1d(z, H_norm, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return {
        'sol': sol,
        'z': z,
        'H_norm': H_norm,
        'Hz_interp': Hz_interp,
        'H0': H0,
        'Omega_m_eff': Omega_m_eff,
        'alpha0': alpha0,
        'alpha_dot0': alpha_dot0
    }

# ============================================================
# 4. 校准 Q=0 分支（精确恢复 ΛCDM）
# ============================================================
print("="*60)
print("步骤1：校准 Q=0 至 ΛCDM (Ω_m=0.3, H0≈1)")
print("="*60)

def objective(rho_m_i):
    res = integrate_to_present(rho_m_i, 0.0)
    if res is None:
        return 1e6
    return res['Omega_m_eff'] - 0.3

rho_m0_nocouple = 0.3 * 3.0
rho_m_i_guess = rho_m0_nocouple * a_i**(-3)

rho_low = rho_m_i_guess * 0.1
rho_high = rho_m_i_guess * 10.0
f_low = objective(rho_low)
f_high = objective(rho_high)
for _ in range(30):
    if f_low * f_high <= 0:
        break
    if abs(f_low) < abs(f_high):
        rho_low /= 1.5
        f_low = objective(rho_low)
    else:
        rho_high *= 1.5
        f_high = objective(rho_high)

result = root_scalar(objective, bracket=[rho_low, rho_high], method='brentq', xtol=1e-6)
rho_m_i_calibrated = result.root
print(f"校准完成！初始物质密度 = {rho_m_i_calibrated:.6e}")
res_Q0 = integrate_to_present(rho_m_i_calibrated, 0.0)
print(f"验证 Q=0: Ω_m_eff = {res_Q0['Omega_m_eff']:.6f}, H0 = {res_Q0['H0']:.6f} (期望 ≈1.0)\n")

# ============================================================
# 5. 扫描 Q 网格，并**将校准的 Q=0 分支显式加入结果集**
# ============================================================
print("步骤2：扫描 Q 网格（约30秒）")
Q_grid = np.linspace(-0.10, 0.15, 30)
Q_valid = []
results_Q = {}

# 首先加入校准的 Q=0 分支
results_Q[0.0] = res_Q0
Q_valid.append(0.0)

for Q in Q_grid:
    # 跳过已存在的0.0（网格中可能没有精确0.0，但以防万一）
    if abs(Q) < 1e-10:
        continue
    print(f"  处理 Q = {Q:+.3f} ...", end='')
    res = integrate_to_present(rho_m_i_calibrated, Q)
    if res is not None and res['z'].max() > 2.0:
        Q_valid.append(Q)
        results_Q[Q] = res
        print(f" 成功 (H0={res['H0']:.3f})")
    else:
        print(" 失败")

Q_valid = sorted(Q_valid)  # 排序
print(f"\n成功积分 {len(Q_valid)} 个 Q 点 (包含 Q=0)\n")

# ============================================================
# 6. 图1：校准验证（Q=0 精确恢复 ΛCDM）
# ============================================================
print("步骤3：生成图1 (校准验证) ...")
z_plot = np.linspace(0, 2.5, 200)
Hz_LCDM = np.sqrt(0.3*(1+z_plot)**3 + 0.7)

plt.figure(figsize=(8,5))
plt.plot(z_plot, Hz_LCDM, 'k--', linewidth=2, label=r'$\Lambda$CDM ($\Omega_m=0.3$)')
Hz_Q0 = results_Q[0.0]['Hz_interp'](z_plot)
plt.plot(z_plot, Hz_Q0, 'r-', linewidth=2, label='CLST $Q=0$ (校准后)')
plt.xlabel(r'Redshift $z$', fontsize=14)
plt.ylabel(r'$H(z)/H_0$', fontsize=14)
plt.title('Figure 1: Calibration – $Q=0$ branch exactly recovers $\\Lambda$CDM', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 2.5); plt.ylim(0.5, 2.0)
plt.tight_layout()
plt.savefig('fig1_calibration.pdf', dpi=300)
plt.close()
print("  已保存: fig1_calibration.pdf")

# ============================================================
# 7. 图2：不同 Q 下的 H(z)/H0（代表性曲线）
# ============================================================
print("步骤4：生成图2 (耦合对膨胀历史的影响) ...")
plt.figure(figsize=(10,6))
plt.plot(z_plot, Hz_LCDM, 'k--', linewidth=2, label=r'$\Lambda$CDM')

# 选取代表性 Q 值（确保 Q=0 存在）
demo_Q = [-0.02, 0.0, 0.02, 0.04]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(demo_Q)))
for i, Q in enumerate(demo_Q):
    if Q in results_Q:
        res = results_Q[Q]
        Hz = res['Hz_interp'](z_plot)
        plt.plot(z_plot, Hz, color=colors[i], linewidth=2,
                 label=rf'$Q={Q:+.2f}$, $\Omega_m^{{\mathrm{{eff}}}}={res["Omega_m_eff"]:.2f}$')

plt.xlabel(r'Redshift $z$', fontsize=14)
plt.ylabel(r'$H(z)/H_0$', fontsize=14)
plt.title('Figure 2: Impact of effective coupling on expansion history', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 2.5); plt.ylim(0.5, 2.0)
plt.tight_layout()
plt.savefig('fig2_H_of_z.pdf', dpi=300)
plt.close()
print("  已保存: fig2_H_of_z.pdf")

# ============================================================
# 8. 图3：H0/H0(Q=0) 与 Q 的关系（修复索引问题）
# ============================================================
print("步骤5：生成图3 (H0 对耦合强度的依赖) ...")
Q_vals = np.array(sorted(results_Q.keys()))
H0_vals = np.array([results_Q[Q]['H0'] for Q in Q_vals])

# 修复：找到最接近 0 的 Q 作为归一化基准（而不是精确相等）
idx_zero = np.argmin(np.abs(Q_vals))
H0_norm = H0_vals / H0_vals[idx_zero]

plt.figure(figsize=(8,5))
plt.plot(Q_vals, H0_norm, 'o-', color='darkblue', markersize=4, linewidth=1.5)
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel(r'Effective coupling $Q$', fontsize=14)
plt.ylabel(r'$H_0 / H_0(Q=0)$', fontsize=14)
plt.title('Figure 3: Today\'s Hubble parameter vs coupling strength', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_H0_vs_Q.pdf', dpi=300)
plt.close()
print("  已保存: fig3_H0_vs_Q.pdf")

# ============================================================
# 9. 生成模拟超新星数据（ΛCDM 真值）
# ============================================================
print("步骤6：生成模拟超新星数据 ...")
np.random.seed(42)
z_data = np.linspace(0.01, 2.3, 50)
H0_true = 70.0
Omega_m_true = 0.3
c_light = 299792.458

def mu_lcdm(z, H0, Omega_m):
    from scipy.integrate import cumulative_trapezoid
    E = np.sqrt(Omega_m*(1+z)**3 + (1-Omega_m))
    integrand = 1/E
    dc = cumulative_trapezoid(integrand, z, initial=0) * c_light / H0
    dl = (1+z) * np.maximum(dc, 1e-10)
    return 5 * np.log10(dl) + 25

mu_true = mu_lcdm(z_data, H0_true, Omega_m_true)
mu_err = 0.15 * np.ones_like(z_data)
mu_obs = mu_true + np.random.randn(len(z_data)) * mu_err
print("  模拟数据生成完成")

# ============================================================
# 10. 构建 MCMC 代理插值器（包含 Q=0）
# ============================================================
print("步骤7：构建 MCMC 代理插值器 ...")
Hz_interp_dict = {Q: results_Q[Q]['Hz_interp'] for Q in Q_vals}
H0_dict = {Q: results_Q[Q]['H0'] for Q in Q_vals}
Q_valid_array = np.array(Q_vals)

def Hz_interp_for_Q(Q, z):
    if Q in Hz_interp_dict:
        return Hz_interp_dict[Q](z)
    if Q < Q_valid_array.min() or Q > Q_valid_array.max():
        return np.full_like(z, np.nan)
    idx = np.searchsorted(Q_valid_array, Q)
    if idx == 0:
        q1, q2 = Q_valid_array[0], Q_valid_array[1]
        f1, f2 = Hz_interp_dict[q1](z), Hz_interp_dict[q2](z)
    elif idx >= len(Q_valid_array):
        q1, q2 = Q_valid_array[-2], Q_valid_array[-1]
        f1, f2 = Hz_interp_dict[q1](z), Hz_interp_dict[q2](z)
    else:
        q1, q2 = Q_valid_array[idx-1], Q_valid_array[idx]
        f1, f2 = Hz_interp_dict[q1](z), Hz_interp_dict[q2](z)
    Hz = f1 + (f2 - f1) * (Q - q1) / (q2 - q1)
    return np.maximum(Hz, 1e-10)

def mu_theory(z, H0, Q):
    Hz = Hz_interp_for_Q(Q, z) * H0
    if np.any(np.isnan(Hz)) or np.any(Hz <= 0):
        return np.inf * np.ones_like(z)
    from scipy.integrate import cumulative_trapezoid
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    Hz_sorted = Hz[sort_idx]
    integrand = 1.0 / Hz_sorted
    dc_sorted = cumulative_trapezoid(integrand, z_sorted, initial=0) * c_light
    if np.any(dc_sorted < 0):
        return np.inf * np.ones_like(z)
    dc = np.zeros_like(z)
    dc[sort_idx] = dc_sorted
    dl = (1 + z) * np.maximum(dc, 1e-10)
    return 5 * np.log10(dl) + 25

# ============================================================
# 11. MCMC 采样
# ============================================================
print("步骤8：运行 MCMC 采样（约20秒）...")
def log_likelihood(theta):
    Q, H0 = theta
    if Q < Q_valid_array.min() or Q > Q_valid_array.max():
        return -np.inf
    if H0 < 50 or H0 > 90:
        return -np.inf
    mu_th = mu_theory(z_data, H0, Q)
    if np.any(np.isinf(mu_th)):
        return -np.inf
    chi2 = np.sum(((mu_th - mu_obs) / mu_err)**2)
    return -0.5 * chi2

def log_prior(theta):
    Q, H0 = theta
    if Q_valid_array.min() <= Q <= Q_valid_array.max() and 50 <= H0 <= 90:
        return 0.0
    return -np.inf

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

ndim = 2
nwalkers = 32
nsteps = 8000
burnin = 2000

initial = np.array([0.02, 70.0])
pos = initial + np.array([0.05, 2.0]) * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, nsteps, progress=True)
samples = sampler.get_chain(discard=burnin, flat=True)

# ============================================================
# 12. 图4：MCMC corner plot
# ============================================================
print("步骤9：生成图4 (MCMC 后验分布) ...")
fig = corner.corner(samples, labels=[r'$Q$', r'$H_0$ (km/s/Mpc)'],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})
fig.savefig('fig4_mcmc_corner.pdf', dpi=200)
plt.close()
print("  已保存: fig4_mcmc_corner.pdf")

# ============================================================
# 13. 输出结果摘要
# ============================================================
Q_mcmc = np.percentile(samples[:,0], [16, 50, 84])
H0_mcmc = np.percentile(samples[:,1], [16, 50, 84])
print("\n" + "="*60)
print("MCMC 参数估计结果")
print("="*60)
print(f"Q  = {Q_mcmc[1]:.4f} +{Q_mcmc[2]-Q_mcmc[1]:.4f}/-{Q_mcmc[1]-Q_mcmc[0]:.4f}")
print(f"H0 = {H0_mcmc[1]:.1f} +{H0_mcmc[2]-H0_mcmc[1]:.1f}/-{H0_mcmc[1]-H0_mcmc[0]:.1f} km/s/Mpc")
print("\n所有图片已生成完毕！")
print("="*60)