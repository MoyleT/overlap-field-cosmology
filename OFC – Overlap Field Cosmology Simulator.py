# ============================================================
# 宇宙图层叠加论 · 重叠场宇宙学模拟器（含宇宙常数 Λ）
# 修正版本：w_eff 正确，m_α 调至 0.6，支持负 Q
# 日期：2026.02.12  (修正版)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# ------------------------------
# 1. 模型参数（已优化）
# ------------------------------
Q_list = [0.0, 0.02, 0.04, -0.02]  # 包含正负耦合
m_alpha = 0.6  # 减小质量，使场更晚滚动，今天α≈0
V_type = 'quadratic'

# 宇宙学常数：Ω_Λ = 0.7 对应的密度（8πG=1, H0≈1 单位）
rho_Lambda = 0.7 * 3.0


# 势能函数
def V(alpha):
    return 0.5 * m_alpha ** 2 * alpha ** 2


def Vp(alpha):
    return m_alpha ** 2 * alpha


# ------------------------------
# 2. ODE 系统与哈勃函数
# ------------------------------
def system(t, y, Q):
    a, alpha, alpha_dot, rho_m = y
    H = Hubble(alpha, alpha_dot, rho_m, Q)
    if H <= 0:
        H = 1e-10
    da_dt = a * H
    dalpha_dt = alpha_dot
    dalpha_dot_dt = -3.0 * H * alpha_dot - Vp(alpha) - Q * rho_m  # 注意负号，与推导一致
    denom = 1.0 - alpha * Q
    if abs(denom) < 1e-10:
        denom = 1e-10
    drho_m_dt = -3.0 * H * rho_m + (2.0 * Q * rho_m * alpha_dot) / denom
    return [da_dt, dalpha_dt, dalpha_dot_dt, drho_m_dt]


def Hubble(alpha, alpha_dot, rho_m, Q):
    rho_eff = (1.0 - alpha * Q) * rho_m + 0.5 * alpha_dot ** 2 + V(alpha) + rho_Lambda
    H2 = (1.0 / 3.0) * rho_eff
    return np.sqrt(max(H2, 0.0))


# ------------------------------
# 3. 固定演化参数与打靶函数
# ------------------------------
a_i = 1e-5
alpha_i = 0.0
alpha_dot_i = 0.0


def integrate_rho_m_i(rho_m_i, Q):
    """给定初始物质密度，积分至 a=1，返回今天 Ω_m_eff 和 H0"""
    t_span = (0.0, 200.0)
    y0 = [a_i, alpha_i, alpha_dot_i, rho_m_i]

    def hit_a1(t, y):
        return y[0] - 1.0

    hit_a1.terminal = True
    hit_a1.direction = 1

    sol = solve_ivp(lambda t, y: system(t, y, Q),
                    t_span, y0, method='DOP853',
                    events=hit_a1, rtol=1e-10, atol=1e-12)
    if not sol.success or len(sol.t_events[0]) == 0:
        return None, None

    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(sol.t))])
    H0 = H[-1]
    Omega_m = (1.0 - alpha[-1] * Q) * rho_m[-1] / (3.0 * H0 ** 2)
    return Omega_m, H0


# ------------------------------
# 4. 校准 Q=0：使 Ω_m=0.3 且 H0≈1
# ------------------------------
print("=== 校准 Q=0 至 ΛCDM (Ω_m=0.3, H0≈1) ===\n")
target_Omega = 0.3

# 粗略初始估计（无耦合时的物质密度）
rho_m0_nocouple = 0.3 * 3.0
rho_m_i_guess = rho_m0_nocouple * a_i ** (-3)


def objective(rho_m_i):
    Omega, H0 = integrate_rho_m_i(rho_m_i, 0.0)
    if Omega is None:
        return 1e6
    return Omega - target_Omega


# 自动扩增搜索区间
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

if f_low * f_high > 0:
    raise RuntimeError("无法找到变号区间，请检查模型参数")

result = root_scalar(objective, bracket=[rho_low, rho_high], method='brentq', xtol=1e-6)
rho_m_i_calibrated = result.root
print(f"校准完成！初始物质密度 = {rho_m_i_calibrated:.6e}")

# 验证 Q=0 结果
Omega0, H0_actual = integrate_rho_m_i(rho_m_i_calibrated, 0.0)
print(f"验证 Q=0: Ω_m_eff = {Omega0:.6f}, H0_actual = {H0_actual:.6f} (期望 ≈1.0)\n")


# ------------------------------
# 5. 扫描 Q 值
# ------------------------------
def run_simulation(Q):
    t_span = (0.0, 200.0)
    y0 = [a_i, alpha_i, alpha_dot_i, rho_m_i_calibrated]

    def hit_a1(t, y):
        return y[0] - 1.0

    hit_a1.terminal = True
    hit_a1.direction = 1

    sol = solve_ivp(lambda t, y: system(t, y, Q),
                    t_span, y0, method='DOP853',
                    events=hit_a1, rtol=1e-10, atol=1e-12,
                    dense_output=True)
    t = sol.t
    a = sol.y[0]
    alpha = sol.y[1]
    alpha_dot = sol.y[2]
    rho_m = sol.y[3]
    H = np.array([Hubble(alpha[i], alpha_dot[i], rho_m[i], Q) for i in range(len(t))])

    H0_actual = H[-1]
    z = 1.0 / a - 1.0
    H_norm = H / H0_actual  # 归一化到 H0=1

    # 今天有效参数
    Omega_m_eff = (1.0 - alpha[-1] * Q) * rho_m[-1] / (3.0 * H0_actual ** 2)

    # --- 修正：状态方程 w_eff 的正确计算 ---
    rho_alpha = 0.5 * alpha_dot ** 2 + V(alpha)
    p_alpha = 0.5 * alpha_dot ** 2 - V(alpha)
    rho_de = rho_alpha + rho_Lambda
    p_de = p_alpha - rho_Lambda
    w_eff = p_de / rho_de
    w_eff_today = w_eff[-1]

    # 插值器
    Hz_interp = interp1d(z, H_norm, kind='cubic', bounds_error=False, fill_value='extrapolate')

    return {
        'Q': Q,
        'z': z,
        'H_norm': H_norm,
        'Omega_m_eff': Omega_m_eff,
        'alpha0': alpha[-1],
        'alpha_dot0': alpha_dot[-1],
        'w_eff0': w_eff_today,
        'H0_actual': H0_actual,
        'Hz_interp': Hz_interp
    }


print("=== 扫描耦合参数 Q ===\n")
results = []
for Q in Q_list:
    print(f"计算 Q = {Q:+.3f} ...")
    res = run_simulation(Q)
    results.append(res)
    print(f"  完成: Ω_m_eff = {res['Omega_m_eff']:.3f}, α0 = {res['alpha0']:.3f}, "
          f"w_eff0 = {res['w_eff0']:.3f}, H0_actual = {res['H0_actual']:.3f}\n")

# ------------------------------
# 6. 绘图：H(z)/H0 对比
# ------------------------------
plt.figure(figsize=(10, 6))
z_plot = np.linspace(0, 2.5, 200)

# ΛCDM 理论曲线（Ω_m=0.3, Ω_Λ=0.7）
Hz_LCDM = np.sqrt(0.3 * (1 + z_plot) ** 3 + 0.7)
plt.plot(z_plot, Hz_LCDM, 'k--', linewidth=2, label=r'$\Lambda$CDM (理论)')

colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(results)))
for i, res in enumerate(results):
    Hz = res['Hz_interp'](z_plot)
    plt.plot(z_plot, Hz, color=colors[i], linewidth=1.8,
             label=rf'Q={res["Q"]:+.2f}, $\Omega_m^{{\mathrm{{eff}}}}$={res["Omega_m_eff"]:.2f}, $w_{{\mathrm{{eff}}}}$={res["w_eff0"]:.2f}')

plt.xlabel('红移 $z$', fontsize=14)
plt.ylabel(r'$H(z)/H_0$', fontsize=14)
plt.title('重叠场耦合对哈勃参数的影响（修正状态方程，$m_\\alpha=0.6$）', fontsize=16)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 2.5)
plt.ylim(0.5, 2.0)
plt.tight_layout()
plt.savefig('layer_coupling_corrected.pdf', dpi=300)
plt.show()

# 可选：绘制 α(z) 演化
plt.figure(figsize=(10, 4))
for i, res in enumerate(results):
    if res['Q'] != 0:
        plt.plot(res['z'], res['alpha0'] * np.ones_like(res['z']), '--')  # 简略，实际应插值
        # 实际 alpha(z) 数据点太多，简单绘制最后值示意
        plt.scatter(0, res['alpha0'], color=colors[i], label=f'Q={res["Q"]:+.2f}')
plt.xlabel('红移 $z$')
plt.ylabel('重叠场 $\\alpha$')
plt.title('今天 $\\alpha$ 值（$m_\\alpha=0.6$）')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 2.5)
plt.show()

print("=== 模拟完成 ===")