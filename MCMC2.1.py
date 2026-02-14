#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇宙图层叠加论（CLST）重叠场模型 · Pantheon+ 超新星数据拟合
增强版：为每个Q单独校准初始密度，加入收敛诊断，支持模拟数据测试
作者：马延亮（代码整理）
日期：2026-02-15

依赖：numpy, scipy, matplotlib, emcee, corner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import emcee
import corner

# =============================================================================
# 1. 模型参数与方程（自然单位制 8πG = 1, c = 1）
# =============================================================================
m_alpha = 0.6                     # 重叠场质量
rho_Lambda = 0.7 * 3.0            # 宇宙常数密度，确保Q=0时Ω_m=0.3, H0=1
a_i = 1e-5                         # 初始尺度因子
alpha_i = 0.0                      # 初始重叠场
alpha_dot_i = 0.0                  # 初始重叠场时间导数

def V(alpha):
    """重叠场势能：二次型"""
    return 0.5 * m_alpha**2 * alpha**2

def Vp(alpha):
    """势能导数"""
    return m_alpha**2 * alpha

def Hubble(alpha, alpha_dot, rho_m, Q):
    """哈勃参数计算（自然单位）"""
    rho_eff = (1.0 - alpha * Q) * rho_m + 0.5 * alpha_dot**2 + V(alpha) + rho_Lambda
    H2 = rho_eff / 3.0
    return np.sqrt(max(H2, 0.0))

def system(t, y, Q):
    """FLRW背景演化方程组"""
    a, alpha, alpha_dot, rho_m = y
    H = Hubble(alpha, alpha_dot, rho_m, Q)
    if H <= 0 or not np.isfinite(H):
        H = 1e-10

    da_dt = a * H
    dalpha_dt = alpha_dot
    dalpha_dot_dt = -3.0 * H * alpha_dot - Vp(alpha) - Q * rho_m

    denom = 1.0 - alpha * Q
    if abs(denom) < 1e-10:
        denom = 1e-10 * np.sign(denom) if denom != 0 else 1e-10
    drho_m_dt = -3.0 * H * rho_m + (2.0 * Q * rho_m * alpha_dot) / denom

    return [da_dt, dalpha_dt, dalpha_dot_dt, drho_m_dt]

# =============================================================================
# 2. 积分辅助函数
# =============================================================================
def integrate_to_a1(Q, rho_m_i, rtol=1e-10, atol=1e-12, t_max=500.0):
    """从初始条件积分至 a=1，返回解对象或None"""
    t_span = (0.0, t_max)
    y0 = [a_i, alpha_i, alpha_dot_i, rho_m_i]
    def hit_a1(t, y):
        return y[0] - 1.0
    hit_a1.terminal = True
    hit_a1.direction = 1

    sol = solve_ivp(lambda t, y: system(t, y, Q),
                    t_span, y0, method='DOP853',
                    events=hit_a1, rtol=rtol, atol=atol,
                    dense_output=True)
    if sol.success and len(sol.t_events[0]) > 0:
        return sol
    return None

def get_Omega0_from_sol(sol, Q):
    """从积分结果提取今天（a=1）的有效Ω_m"""
    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(sol.t))])
    Omega = (1.0 - alpha[-1]*Q) * rho_m[-1] / (3.0 * H[-1]**2)
    return Omega

# =============================================================================
# 3. 打靶法校准：对任意 Q，找到使今天 Ω_m=0.3 的初始物质密度
# =============================================================================
def calibrate_rho_m_i_for_Q(Q, rho_guess, target_Omega=0.3):
    """打靶法：调整初始物质密度，使今天有效Ω_m达到 target_Omega"""
    def objective(rho_m_i):
        sol = integrate_to_a1(Q, rho_m_i, rtol=1e-10, atol=1e-12)
        if sol is None:
            return 1e6
        Omega = get_Omega0_from_sol(sol, Q)
        return Omega - target_Omega

    # 自动扩增搜索区间
    rho_low = rho_guess * 0.1
    rho_high = rho_guess * 10.0
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
    else:
        raise RuntimeError(f"无法为 Q={Q} 找到变号区间")

    res = root_scalar(objective, bracket=[rho_low, rho_high], method='brentq', xtol=1e-6)
    return res.root

# =============================================================================
# 4. 加载真实超新星数据（Pantheon_SHOES.dat）
# =============================================================================
def load_supernova_data(filename, skip_header=1, z_col=2, mu_col=10, err_col=11):
    """
    加载超新星数据文件，支持跳过标题行并选择指定列。
    默认参数适用于 Pantheon_SHOES.dat：
        - 红移列：zHD (索引2)
        - 距离模数列：MU_SH0ES (索引10)
        - 误差列：MU_SH0ES_ERR_DIAG (索引11)
    自动过滤无效值（如 -9, -999）。
    """
    data = np.genfromtxt(filename, skip_header=skip_header,
                         usecols=(z_col, mu_col, err_col),
                         dtype=float, encoding='utf-8')
    z = data[:, 0]
    mu = data[:, 1]
    err = data[:, 2]

    valid = (mu > -90) & (err > 0) & (np.isfinite(z)) & (np.isfinite(mu)) & (np.isfinite(err))
    z = z[valid]
    mu = mu[valid]
    err = err[valid]

    sort_idx = np.argsort(z)
    z = z[sort_idx]
    mu = mu[sort_idx]
    err = err[sort_idx]

    print(f"成功加载 {len(z)} 条超新星数据，红移范围 [{z.min():.3f}, {z.max():.3f}]")
    return z, mu, err

# =============================================================================
# 5. 生成模拟超新星数据（用于测试）
# =============================================================================
def generate_simulated_data():
    np.random.seed(42)
    z = np.linspace(0.01, 2.3, 50)
    H0_true = 70.0
    Omega_m_true = 0.3
    c = 299792.458
    from scipy.integrate import cumulative_trapezoid
    E = np.sqrt(Omega_m_true * (1+z)**3 + (1 - Omega_m_true))
    integrand = 1/E
    dc = cumulative_trapezoid(integrand, z, initial=0) * c / H0_true
    dl = (1 + z) * dc
    mu_true = 5 * np.log10(dl) + 25
    mu_err = 0.15 * np.ones_like(z)
    mu_obs = mu_true + np.random.randn(len(z)) * mu_err
    print(f"生成 {len(z)} 条模拟超新星数据（真值 H0={H0_true}, Ω_m={Omega_m_true})")
    return z, mu_obs, mu_err

# =============================================================================
# 6. 预计算 Q 网格与插值器（每个Q独立校准）
# =============================================================================
def build_Q_interpolators(rho_m_i_Q0, Q_min=-0.10, Q_max=0.15, n_points=30):
    """
    扫描Q值，对每个Q进行单独校准，构建 H(z)/H0 的插值器
    返回：
        Q_valid : list, 成功积分的Q值
        Hz_interp_dict : dict, 每个Q对应的插值函数
        H0_dict : dict, 每个Q对应的实际H0值（自然单位）
    """
    Q_grid = np.linspace(Q_min, Q_max, n_points)
    Q_valid = []
    Hz_interp_dict = {}
    H0_dict = {}

    # 首先处理 Q=0，用已校准的初始密度
    print("  处理 Q = 0.000 (使用已校准初值) ...", end='')
    sol = integrate_to_a1(0.0, rho_m_i_Q0, rtol=1e-10, atol=1e-12)
    if sol is None:
        raise RuntimeError("Q=0 积分失败，无法继续")
    t = sol.t
    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], 0.0) for i in range(len(t))])
    H0_act = H[-1]
    z = 1.0 / a - 1.0
    H_norm = H / H0_act
    interp = interp1d(z, H_norm, kind='cubic', bounds_error=False, fill_value='extrapolate')
    Q_valid.append(0.0)
    Hz_interp_dict[0.0] = interp
    H0_dict[0.0] = H0_act
    print(f" 成功 (H0={H0_act:.3f})")

    # 处理其他Q，单独校准
    for Q in Q_grid:
        if abs(Q) < 1e-10:
            continue
        print(f"  处理 Q = {Q:+.3f} ...", end='')
        try:
            # 以Q=0的初值为起点进行校准
            rho_m_i_calib = calibrate_rho_m_i_for_Q(Q, rho_m_i_Q0, target_Omega=0.3)
        except Exception as e:
            print(f" 校准失败: {e}")
            continue

        sol = integrate_to_a1(Q, rho_m_i_calib, rtol=1e-10, atol=1e-12)
        if sol is None:
            print(" 积分失败")
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
        interp = interp1d(z, H_norm, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Q_valid.append(Q)
        Hz_interp_dict[Q] = interp
        H0_dict[Q] = H0_act
        print(f" 成功 (H0={H0_act:.3f})")

    if len(Q_valid) == 0:
        raise RuntimeError("没有成功积分的 Q 值")
    print(f"\n成功构建 {len(Q_valid)} 个 Q 网格点: {np.round(Q_valid,3)}")
    return Q_valid, Hz_interp_dict, H0_dict

# =============================================================================
# 7. 理论距离模数计算（使用插值器）
# =============================================================================
c_light = 299792.458  # km/s，用于将H0转换为km/s/Mpc

def mu_theory(z, H0, Q, Q_valid, Hz_interp_dict):
    """
    计算给定红移数组的理论距离模数
    H0 单位：km/s/Mpc
    Q 必须位于 Q_valid 范围内
    """
    Q_valid = np.asarray(Q_valid)
    if Q < Q_valid.min() or Q > Q_valid.max():
        return np.full_like(z, np.inf)

    # 对任意Q，在网格点间线性插值得到 H(z)/H0
    z_arr = np.asarray(z)
    Q_arr = Q_valid
    idx = np.searchsorted(Q_arr, Q)
    if idx == 0:
        q1, q2 = Q_arr[0], Q_arr[1]
        f1 = Hz_interp_dict[q1](z_arr)
        f2 = Hz_interp_dict[q2](z_arr)
    elif idx >= len(Q_arr):
        q1, q2 = Q_arr[-2], Q_arr[-1]
        f1 = Hz_interp_dict[q1](z_arr)
        f2 = Hz_interp_dict[q2](z_arr)
    else:
        q1, q2 = Q_arr[idx-1], Q_arr[idx]
        f1 = Hz_interp_dict[q1](z_arr)
        f2 = Hz_interp_dict[q2](z_arr)
    Hz_ratio = f1 + (f2 - f1) * (Q - q1) / (q2 - q1)
    Hz_ratio = np.maximum(Hz_ratio, 1e-10)

    Hz = Hz_ratio * H0

    # 积分求光度距离
    from scipy.integrate import cumulative_trapezoid
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    Hz_sorted = Hz[sort_idx]

    integrand = 1.0 / Hz_sorted
    dc_sorted = cumulative_trapezoid(integrand, z_sorted, initial=0) * c_light
    if np.any(dc_sorted < 0):
        return np.inf * np.ones_like(z_arr)

    dc = np.zeros_like(z_arr)
    dc[sort_idx] = dc_sorted
    dl = (1 + z_arr) * dc
    dl = np.maximum(dl, 1e-10)
    mu = 5 * np.log10(dl) + 25
    return mu

# =============================================================================
# 8. MCMC 参数估计与收敛诊断
# =============================================================================
def gelman_rubin(samples):
    """
    计算 Gelman-Rubin 统计量 R_hat
    samples 形状为 (n_chains, n_steps, n_dim)
    """
    n_chains, n_steps, n_dim = samples.shape
    chain_mean = np.mean(samples, axis=1)          # (n_chains, n_dim)
    overall_mean = np.mean(chain_mean, axis=0)    # (n_dim,)
    # 链间方差 B/n
    B = n_steps * np.var(chain_mean, axis=0, ddof=1)
    # 链内方差 W
    W = np.mean(np.var(samples, axis=1, ddof=1), axis=0)
    # 估计边缘后验方差
    var_plus = (n_steps - 1) / n_steps * W + (1 / n_steps) * B
    R_hat = np.sqrt(var_plus / W)
    return R_hat

def run_mcmc(z_data, mu_data, mu_err, Q_valid, Hz_interp_dict,
             Q_prior_range=None, H0_prior_range=(50, 90),
             nwalkers=32, nsteps=8000, burnin=2000):
    """
    运行 MCMC 采样，返回 samples (flat) 和 sampler 对象
    """
    if Q_prior_range is None:
        Q_prior_range = (min(Q_valid), max(Q_valid))

    ndim = 2

    def log_likelihood(theta):
        Q, H0 = theta
        if Q < Q_prior_range[0] or Q > Q_prior_range[1] or H0 < H0_prior_range[0] or H0 > H0_prior_range[1]:
            return -np.inf
        mu_th = mu_theory(z_data, H0, Q, Q_valid, Hz_interp_dict)
        if np.any(np.isinf(mu_th)):
            return -np.inf
        chi2 = np.sum(((mu_th - mu_data) / mu_err)**2)
        return -0.5 * chi2

    def log_prior(theta):
        Q, H0 = theta
        if Q_prior_range[0] <= Q <= Q_prior_range[1] and H0_prior_range[0] <= H0 <= H0_prior_range[1]:
            return 0.0
        return -np.inf

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    initial = np.array([0.0, 70.0])
    pos = initial + np.array([0.05, 2.0]) * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    print("\n=== 开始 MCMC 采样 ===")
    sampler.run_mcmc(pos, nsteps, progress=True)

    # 接受率
    acceptance = sampler.acceptance_fraction
    print(f"平均接受率: {np.mean(acceptance):.3f}")

    # 迹图
    fig, axes = plt.subplots(2, figsize=(10,6), sharex=True)
    samples_chain = sampler.get_chain()
    for i in range(ndim):
        axes[i].plot(samples_chain[:, :, i], alpha=0.3)
        axes[i].set_ylabel(['Q', r'H0'][i])
    axes[-1].set_xlabel('步数')
    plt.tight_layout()
    plt.savefig('trace_plot.pdf')
    plt.show()

    # Gelman-Rubin 诊断
    R_hat = gelman_rubin(samples_chain)
    print(f"Gelman-Rubin R_hat: Q={R_hat[0]:.3f}, H0={R_hat[1]:.3f} (小于1.1表示收敛)")

    samples = sampler.get_chain(discard=burnin, flat=True)
    return samples, sampler

# =============================================================================
# 9. 主程序
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("宇宙图层叠加论 · 重叠场宇宙学 MCMC 拟合（增强版）")
    print("=" * 60)

    # 配置开关：True 使用模拟数据，False 使用真实数据
    use_simulated = False   # 请根据实际情况设置

    # 获取数据
    if use_simulated:
        print("\n[1] 使用模拟数据...")
        z_data, mu_data, mu_err = generate_simulated_data()
    else:
        print("\n[1] 加载真实数据...")
        data_file = "Pantheon_SHOES.dat"
        skip_header = 1
        z_col = 2
        mu_col = 10
        err_col = 11
        try:
            z_data, mu_data, mu_err = load_supernova_data(data_file,
                                                           skip_header=skip_header,
                                                           z_col=z_col,
                                                           mu_col=mu_col,
                                                           err_col=err_col)
        except FileNotFoundError:
            print(f"错误：找不到文件 {data_file}，请确认路径")
            exit(1)
        except Exception as e:
            print(f"数据加载失败：{e}")
            exit(1)

    # 校准 Q=0
    print("\n[2] 校准 Q=0 分支至 ΛCDM (Ω_m=0.3, H0=1)")
    # 利用 calibrate_rho_m_i_for_Q 得到 Q=0 的初始密度
    rho_m_i_Q0 = calibrate_rho_m_i_for_Q(0.0, 9.0e14, target_Omega=0.3)
    print(f"    初始物质密度 ρ_m(a_i) = {rho_m_i_Q0:.3e}")

    # 验证 Q=0
    sol = integrate_to_a1(0.0, rho_m_i_Q0)
    Omega0 = get_Omega0_from_sol(sol, 0.0)
    print(f"    验证 Q=0: Ω_m_eff = {Omega0:.6f}")

    # 预计算 Q 网格
    print("\n[3] 扫描有效耦合参数 Q 并构建插值器...")
    Q_valid, Hz_interp_dict, H0_dict = build_Q_interpolators(rho_m_i_Q0,
                                                              Q_min=-0.10, Q_max=0.15,
                                                              n_points=30)

    # 可选：绘制几个代表性Q的 H(z) 曲线与ΛCDM对比
    z_plot = np.linspace(0.01, 2.5, 100)
    plt.figure(figsize=(8,5))
    for Q in [0.0, 0.04, -0.04]:
        if Q in Hz_interp_dict:
            Hz = Hz_interp_dict[Q](z_plot)
            plt.plot(z_plot, Hz, label=f'Q={Q:+.2f}')
    Hz_lcdm = np.sqrt(0.3*(1+z_plot)**3 + 0.7)
    plt.plot(z_plot, Hz_lcdm, 'k--', label='ΛCDM')
    plt.xlabel('z')
    plt.ylabel('H(z)/H0')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('模型预测的膨胀历史')
    plt.tight_layout()
    plt.savefig('Hz_comparison.pdf')
    plt.show()

    # MCMC 采样
    print("\n[4] 运行 MCMC 采样...")
    samples, sampler = run_mcmc(z_data, mu_data, mu_err,
                                Q_valid, Hz_interp_dict,
                                Q_prior_range=(-0.10, 0.15),
                                H0_prior_range=(50, 90),
                                nwalkers=32, nsteps=8000, burnin=2000)

    # 结果统计
    Q_mcmc = np.percentile(samples[:,0], [16, 50, 84])
    H0_mcmc = np.percentile(samples[:,1], [16, 50, 84])

    print("\n=== MCMC 最终结果 ===")
    print(f"Q  = {Q_mcmc[1]:.4f} +{Q_mcmc[2]-Q_mcmc[1]:.4f}/-{Q_mcmc[1]-Q_mcmc[0]:.4f}")
    print(f"H0 = {H0_mcmc[1]:.1f} +{H0_mcmc[2]-H0_mcmc[1]:.1f}/-{H0_mcmc[1]-H0_mcmc[0]:.1f}")

    # 哈勃张力缓解
    H0_planck = 67.4
    H0_SH0ES = 73.0
    tension_original = H0_SH0ES - H0_planck
    tension_model = H0_mcmc[1] - H0_planck
    relief = (tension_original - tension_model) / tension_original * 100
    print(f"\n哈勃张力缓解程度: {relief:.1f}% (模型预测 H0={H0_mcmc[1]:.1f})")

    # corner plot
    fig = corner.corner(samples, labels=['Q', r'H$_0$ (km/s/Mpc)'],
                        quantiles=[0.16, 0.5, 0.84], bins=50,
                        show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('mcmc_corner.pdf', dpi=200)
    plt.show()

    # 保存链
    np.savetxt('mcmc_chain.txt', samples, header='Q H0')
    print("\n链已保存至 mcmc_chain.txt")
    print("=== 运行完成 ===")