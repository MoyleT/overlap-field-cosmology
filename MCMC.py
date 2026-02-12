# ============================================================
# 宇宙图层叠加论 · MCMC 拟合（最终健壮版）
# 功能：自动校准 Q 网格，生成模拟 ΛCDM 数据，稳健采样
# 作者：马延亮（数值实现）
# 日期：2026.02.14（H0展宽修复）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import emcee
import corner

# ------------------------------
# 1. 固定模型参数
# ------------------------------
m_alpha = 0.6
rho_Lambda = 0.7 * 3.0
a_i = 1e-5
alpha_i = 0.0
alpha_dot_i = 0.0

def V(alpha): return 0.5 * m_alpha**2 * alpha**2
def Vp(alpha): return m_alpha**2 * alpha

# ------------------------------
# 2. ODE 系统（刚性求解器）
# ------------------------------
def system(t, y, Q):
    a, alpha, alpha_dot, rho_m = y
    H = Hubble(alpha, alpha_dot, rho_m, Q)
    if H <= 0 or not np.isfinite(H):
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
    if H2 < 0:
        H2 = 0.0
    return np.sqrt(H2)

# ------------------------------
# 3. 为给定 Q 找到能积分到 a=1 的初始密度
# ------------------------------
def find_working_rho_m_i(Q, rho_guess, tol=0.2, max_trials=5):
    def try_rho(rho_m_i):
        t_span = (0.0, 500.0)
        y0 = [a_i, alpha_i, alpha_dot_i, rho_m_i]
        def hit_a1(t, y): return y[0] - 1.0
        hit_a1.terminal = True
        hit_a1.direction = 1
        sol = solve_ivp(lambda t, y: system(t, y, Q),
                        t_span, y0, method='Radau',
                        events=hit_a1, rtol=1e-8, atol=1e-10,
                        dense_output=True)
        if sol.success and len(sol.t_events[0]) > 0:
            a = sol.y[0]
            alpha = sol.y[1]
            alpha_dot = sol.y[2]
            rho_m = sol.y[3]
            H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(sol.t))])
            H0 = H[-1]
            return sol, H0
        return None, None

    sol, H0 = try_rho(rho_guess)
    if sol is not None:
        return rho_guess, H0, sol

    factors = np.linspace(1 - tol, 1 + tol, max_trials * 2)
    for fac in factors:
        rho_test = rho_guess * fac
        sol, H0 = try_rho(rho_test)
        if sol is not None:
            return rho_test, H0, sol
    return None, None, None

# ------------------------------
# 4. 校准 Q=0 并构建 Q 网格
# ------------------------------
print("=== 预计算 Q 网格（自动校准）===")

def calibrate_Q0():
    target_Omega = 0.3
    rho_m0_nocouple = 0.3 * 3.0
    rho_m_i_guess = rho_m0_nocouple * a_i**(-3)
    def objective(rho_m_i):
        rho, H0, sol = find_working_rho_m_i(0.0, rho_m_i, tol=0.01, max_trials=3)
        if rho is None:
            return 1e6
        a = sol.y[0]
        alpha = sol.y[1]
        alpha_dot = sol.y[2]
        rho_m = sol.y[3]
        H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], 0.0) for i in range(len(sol.t))])
        Omega = (1.0 - alpha[-1]*0.0) * rho_m[-1] / (3.0 * H[-1]**2)
        return Omega - target_Omega
    rho_low = rho_m_i_guess * 0.1
    rho_high = rho_m_i_guess * 10.0
    f_low = objective(rho_low)
    f_high = objective(rho_high)
    for _ in range(30):
        if f_low * f_high <= 0: break
        if abs(f_low) < abs(f_high):
            rho_low /= 1.5; f_low = objective(rho_low)
        else:
            rho_high *= 1.5; f_high = objective(rho_high)
    res = root_scalar(objective, bracket=[rho_low, rho_high], method='brentq', xtol=1e-6)
    return res.root

rho_m_i_Q0 = calibrate_Q0()
print(f"Q=0 校准初始密度: {rho_m_i_Q0:.3e}\n")

Q_grid_raw = np.linspace(-0.1, 0.15, 30)
Q_valid = []
Hz_interp_dict = {}
H0_dict = {}

for Q in Q_grid_raw:
    print(f"  处理 Q = {Q:+.3f} ...", end='')
    rho, H0, sol = find_working_rho_m_i(Q, rho_m_i_Q0, tol=0.2, max_trials=5)
    if rho is None:
        print(" 失败")
        continue
    t = sol.t
    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(t))])
    H0_act = H[-1]
    z = 1.0 / a - 1.0
    if z.max() < 2.0:
        print(f" 红移范围不足 (max z={z.max():.2f})，跳过")
        continue
    H_norm = H / H0_act
    interp = interp1d(z, H_norm, kind='linear', bounds_error=False, fill_value='extrapolate')
    Q_valid.append(Q)
    Hz_interp_dict[Q] = interp
    H0_dict[Q] = H0_act
    print(f" 成功 (H0={H0_act:.3f})")

if len(Q_valid) == 0:
    raise RuntimeError("没有成功积分的 Q 值")
print(f"\n成功构建 {len(Q_valid)} 个 Q 网格点: {np.round(Q_valid,3)}\n")

# ------------------------------
# 5. 生成模拟 ΛCDM 超新星数据（真值 Q=0, H0=70, Ω_m=0.3）
# ------------------------------
def generate_simulated_data():
    np.random.seed(42)
    z = np.linspace(0.01, 2.3, 50)
    H0_true = 70.0
    Omega_m_true = 0.3
    c = 299792.458
    from scipy.integrate import cumulative_trapezoid
    E = np.sqrt(Omega_m_true*(1+z)**3 + (1-Omega_m_true))
    integrand = 1/E
    dc = cumulative_trapezoid(integrand, z, initial=0) * c / H0_true
    dl = (1 + z) * dc
    # 避免 log10(0) 警告
    dl = np.maximum(dl, 1e-10)
    mu_true = 5 * np.log10(dl) + 25
    mu_err = 0.15 * np.ones_like(z)
    mu_obs = mu_true + np.random.randn(len(z)) * mu_err
    print(f"生成 {len(z)} 条模拟超新星数据（真值 Q=0, H0={H0_true}, Ω_m={Omega_m_true})")
    return z, mu_obs, mu_err

z_data, mu_data, mu_err = generate_simulated_data()

# ------------------------------
# 6. 快速插值器：任意 Q 的 H(z)/H0（线性插值，强制非负）
# ------------------------------
Q_valid_array = np.array(Q_valid)

def Hz_interp_for_Q(Q, z):
    if Q in Hz_interp_dict:
        Hz = Hz_interp_dict[Q](z)
    else:
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
    # 强制非负
    Hz = np.maximum(Hz, 1e-10)
    return Hz

# ------------------------------
# 7. 理论距离模数（完全健壮版）
# ------------------------------
c_light = 299792.458

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
    dl = (1 + z) * dc
    # 避免 log10(0)
    dl = np.maximum(dl, 1e-10)
    return 5 * np.log10(dl) + 25

# ------------------------------
# 8. MCMC 参数估计（自由参数 Q, H0）
# ------------------------------
def log_likelihood(theta):
    Q, H0 = theta
    if Q < Q_valid_array.min() or Q > Q_valid_array.max():
        return -np.inf
    if H0 < 50 or H0 > 90:
        return -np.inf
    mu_th = mu_theory(z_data, H0, Q)
    if np.any(np.isinf(mu_th)):
        return -np.inf
    chi2 = np.sum(((mu_th - mu_data) / mu_err)**2)
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

# 初始化采样器
ndim = 2
nwalkers = 32
nsteps = 8000          # 增加步数确保混合
burnin = 2000

initial = np.array([0.02, 70.0])
# ★★★ 大幅增大 H0 的初始散布 ★★★
pos = initial + np.array([0.05, 2.0]) * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
print("\n=== 开始 MCMC 采样 ===")
sampler.run_mcmc(pos, nsteps, progress=True)

samples = sampler.get_chain(discard=burnin, flat=True)

# ------------------------------
# 9. 结果可视化与输出
# ------------------------------
fig = corner.corner(samples, labels=['Q', r'H0 (km/s/Mpc)'],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('mcmc_corner.pdf', dpi=200)
plt.show()

Q_mcmc = np.percentile(samples[:,0], [16, 50, 84])
H0_mcmc = np.percentile(samples[:,1], [16, 50, 84])

print("\n=== MCMC 结果 ===")
print(f"Q  = {Q_mcmc[1]:.4f} +{Q_mcmc[2]-Q_mcmc[1]:.4f}/-{Q_mcmc[1]-Q_mcmc[0]:.4f}")
print(f"H0 = {H0_mcmc[1]:.1f} +{H0_mcmc[2]-H0_mcmc[1]:.1f}/-{H0_mcmc[1]-H0_mcmc[0]:.1f}")

# 哈勃张力缓解（仅为演示，真实数据时调整）
H0_planck = 67.4
H0_SH0ES = 73.0
tension_original = H0_SH0ES - H0_planck
tension_model = H0_mcmc[1] - H0_planck
relief = (tension_original - tension_model) / tension_original * 100
print(f"\n哈勃张力缓解程度: {relief:.1f}%")
print(f"(Planck 2018: {H0_planck}, SH0ES: {H0_SH0ES}, 模型预测: {H0_mcmc[1]:.1f})")

# 保存链
np.savetxt('mcmc_chain.txt', samples)
print("\n=== 模拟完成 ===")