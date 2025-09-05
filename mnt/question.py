# -*- coding: utf-8 -*-
"""
B题·第二问（按B2.docx实现）
多角度联合非线性最小二乘：差分进化 → LM 精炼
模型：
  n(λ) = A + B/λ^2 + C/λ^4               # Cauchy（λ: μm）
  R(σ) ≈ R1 + R2' + 2*sqrt(R1*R2')*cos(Δφ)
  Δφ = 4π n(λ) d cosθ1 * σ               # σ: cm^-1,  d: cm（内部换算）
  R2' = R12 * (1 - R1)^2 * exp(-α d_um)  # α: μm^-1，d_um: μm（外参更直观）
  R1、R12 采用菲涅尔反射（未偏振，s/p 平均）
数据：
  E:\mnt\data\附件1.xlsx  (10 deg)
  E:\mnt\data\附件2.xlsx  (15 deg)
输出：
  拟合参数[d, A, B, C, alpha]、两角残差、图表与CSV
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution, least_squares
from numpy.fft import rfft, rfftfreq

# ---------- Windows 控制台：避免中文/符号编码问题（打印仍用 ASCII） ----------
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans", "Noto Sans CJK SC", "Arial Unicode MS"
]

# ---------------- 基本配置 ----------------
BASE_DIR = r"E:\mnt\data"
FILE_10 = "附件1.xlsx"     # 10 度
FILE_15 = "附件2.xlsx"     # 15 度
THETA0S = [np.deg2rad(10.0), np.deg2rad(15.0)]

# 预处理参数
SG_WIN = 11                 # Savitzky–Golay 平滑窗口（奇数）
SG_POLY = 2
BG_WIN = 151                # 慢变基线滑窗（奇数）

# FFT 初值估计
N_AVG_FOR_FFT = 2.6         # 用于 FFT 估计 d0 的平均折射率
THETA_FOR_FFT = THETA0S[0]  # 用 10 度数据做 FFT

# 差分进化 + LM 边界（B2.docx建议）
A_BOUNDS = (2.4, 2.8)
B_BOUNDS = (-1e4, 1e4)
C_BOUNDS = (-1e8, 1e8)
ALPHA_BOUNDS = (0.0, 0.1)   # μm^-1

# 固定的“有效”基底折射率微偏移，用来产生非零的 R12（不额外引参）
# （外延层与衬底材料几乎相同，但实际存在极小差异；此处给一个固定微偏移等效）
EFF_NS_OFFSET = 0.02

# 输出
CSV_PATH = os.path.join(BASE_DIR, "Q2_jointfit_results.csv")

# ---------------- 工具函数：读取/预处理 ----------------
def read_two_cols_xlsx(path):
    df = pd.read_excel(path, header=None, engine="openpyxl")
    if isinstance(df.iloc[0,0], str):
        df = df.iloc[1:].copy()
    df.columns = ["wavenumber", "reflectance"]
    df["wavenumber"]  = pd.to_numeric(df["wavenumber"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")
    df = df.dropna().sort_values("wavenumber").reset_index(drop=True)
    if df["reflectance"].max() > 1.5:  # 百分比 -> 0~1
        df["reflectance"] = df["reflectance"]/100.0
    return df

def moving_average(x, win):
    if win < 3: return x.copy()
    if win % 2 == 0: win += 1
    k = np.ones(win)/win
    return np.convolve(x, k, mode="same")

def preprocess(wn, R):
    """轻度平滑 + 去慢变基线；返回 y(σ)（条纹信号）和平滑后的 R_smooth（用于拟合）"""
    R_sg = savgol_filter(R, SG_WIN, SG_POLY, mode="interp")
    bg = moving_average(R_sg, BG_WIN)
    y = R_sg - bg
    R_smooth = R_sg  # 拟合用平滑后的曲线（也可直接拟合 R_sg 或 R）
    return y, R_smooth

# ---------------- 光学：Cauchy + Snell + Fresnel ----------------
def n_cauchy(lambda_um, A, B, C):
    return A + B/(lambda_um**2) + C/(lambda_um**4)

def cos_theta_in(n1, theta0, n0=1.0):
    s = np.sin(theta0) * n0 / n1
    # 超临界保护
    s = np.clip(s, 0.0, 0.999999)
    return np.sqrt(1.0 - s*s)

def fresnel_unpolarized_R(n_in, n_out, theta_in):
    """未偏振 R = (|r_s|^2 + |r_p|^2)/2 ；输入介质角 theta_in"""
    cos_in = np.cos(theta_in)
    # Snell 到出射角
    s = np.sin(theta_in) * n_in / n_out
    s = np.clip(s, 0.0, 0.999999)
    cos_out = np.sqrt(1.0 - s*s)
    rs = (n_in*cos_in - n_out*cos_out) / (n_in*cos_in + n_out*cos_out)
    rp = (n_out*cos_in - n_in*cos_out) / (n_out*cos_in + n_in*cos_out)
    return 0.5*(rs*rs + rp*rp)

# ---------------- 模型：两束干涉 + 指数衰减 ----------------
def model_reflectance(wn, theta0, params):
    """
    wn: 波数数组 cm^-1
    theta0: 外部入射角（弧度）
    params: [d_um, A, B, C, alpha_um_inv]
    """
    d_um, A, B, C, alpha = params
    d_cm = d_um * 1e-4
    lam_um = 1e4 / wn
    n1 = n_cauchy(lam_um, A, B, C)          # 膜层折射率
    cos_t1 = cos_theta_in(n1, theta0, 1.0)  # 膜内折射角的 cos

    # 界面反射：
    n0 = 1.0                                # 空气
    n2 = n1 + EFF_NS_OFFSET                 # 有效衬底折射率（微偏移）
    # 上表面（空气-膜）反射
    R1 = fresnel_unpolarized_R(n0, n1, theta0)
    # 下表面（膜-衬底）反射（入射角是膜内角度）
    theta1 = np.arccos(cos_t1)
    R12 = fresnel_unpolarized_R(n1, n2, theta1)

    # 相位与第二束强度（按B2的等效）
    dphi = 4.0*np.pi*n1*cos_t1*wn*d_cm      # 相位差
    R2p = R12 * (1.0 - R1)**2 * np.exp(-alpha * d_um)  # 有效第二束强度

    # 两束叠加（未偏振近似，忽略多次高阶项）
    R_model = R1 + R2p + 2.0*np.sqrt(R1*R2p)*np.cos(dphi)
    # 物理裁剪
    return np.clip(R_model, 0.0, 1.0)

# ---------------- 残差与目标函数（两角联合） ----------------
def residuals_joint(p, data_list, weights=None):
    """
    p: [d_um, A, B, C, alpha]
    data_list: [(wn10, R10, theta0_10), (wn15, R15, theta0_15)]
    weights: 对不同角度/点的权重（可不设）
    """
    d_um, A, B, C, alpha = p
    res_all = []
    for i, (wn, Rexp, theta0) in enumerate(data_list):
        Rpred = model_reflectance(wn, theta0, p)
        res = (Rpred - Rexp)
        if weights is not None:
            res = res * weights[i]
        res_all.append(res)
    return np.concatenate(res_all)

def sse_of_p(p, data_list):
    r = residuals_joint(p, data_list)
    return float(np.dot(r, r))

# ---------------- FFT 估计 d0（用 10° 数据） ----------------
def estimate_d0_via_fft(wn, R, n_avg=2.6, theta0=THETA_FOR_FFT):
    """
    思路：对去趋势后的条纹信号 y(σ) 做 FFT，找到主峰频率 f_peak（cycles per cm^-1）
    对 cos(4π n d cosθ1 σ) 来说，频率 f = 2 n d cosθ1
    => d(cm) = f_peak / (2 n cosθ1)
    """
    # 需要等间距 σ 采样；如不等距，这里做线性插值到等距网格
    wn_sorted = np.sort(wn)
    d_sigma = np.median(np.diff(wn_sorted))
    grid = np.arange(wn_sorted[0], wn_sorted[-1]+d_sigma, d_sigma)
    R_interp = np.interp(grid, wn, R)
    # 轻度平滑 + 去基线
    R_sg = savgol_filter(R_interp, SG_WIN, SG_POLY, mode="interp")
    bg = moving_average(R_sg, BG_WIN)
    y = R_sg - bg
    # FFT
    Y = np.abs(rfft(y))
    freqs = rfftfreq(len(y), d=d_sigma)  # cycles per cm^-1
    # 去掉直流与极低频，找主峰
    mask = freqs > 0.0
    if np.sum(mask) < 3:
        return 5.0  # 兜底一个数（μm）
    fp = freqs[mask][np.argmax(Y[mask])]
    # 估计厚度（cm）
    cos_t1 = cos_theta_in(n_avg, theta0, 1.0)
    d_cm = fp / (2.0 * n_avg * cos_t1)
    d_um = d_cm * 1e4
    return float(np.clip(d_um, 0.1, 200.0))  # 合理范围内

# ---------------- 主流程 ----------------
def main():
    # 读取数据
    path10 = os.path.join(BASE_DIR, FILE_10)
    path15 = os.path.join(BASE_DIR, FILE_15)
    if not os.path.exists(path10) or not os.path.exists(path15):
        raise FileNotFoundError("missing files: E:\\mnt\\data\\附件1.xlsx and 附件2.xlsx")

    df10 = read_two_cols_xlsx(path10)
    df15 = read_two_cols_xlsx(path15)

    # 预处理（平滑 + 去基线）— 拟合仍用平滑后的曲线
    y10, R10 = preprocess(df10["wavenumber"].values, df10["reflectance"].values)
    y15, R15 = preprocess(df15["wavenumber"].values, df15["reflectance"].values)
    wn10 = df10["wavenumber"].values.astype(float)
    wn15 = df15["wavenumber"].values.astype(float)

    # FFT 估计 d0（用 10°）
    d0_um = estimate_d0_via_fft(wn10, df10["reflectance"].values, n_avg=N_AVG_FOR_FFT, theta0=THETA_FOR_FFT)
    d_bounds = (0.8*d0_um, 1.2*d0_um)

    print("== Initial guess via FFT ==")
    print(f"estimated d0 (um) ~ {d0_um:.2f}, search bounds: [{d_bounds[0]:.2f}, {d_bounds[1]:.2f}]")

    # 组装联合数据
    data_list = [
        (wn10, R10, THETA0S[0]),
        (wn15, R15, THETA0S[1]),
    ]

    # 差分进化（全局）
    bounds = [d_bounds, A_BOUNDS, B_BOUNDS, C_BOUNDS, ALPHA_BOUNDS]
    def fun_sse(p): return sse_of_p(p, data_list)
    print("== Differential Evolution (global search) ==")
    result_de = differential_evolution(fun_sse, bounds=bounds, strategy="best1bin",
                                       maxiter=200, popsize=30, tol=1e-6, polish=False, seed=42)
    p_de = result_de.x
    print("DE best (d, A, B, C, alpha):", [f"{v:.6g}" for v in p_de], "  SSE:", f"{result_de.fun:.6g}")

    # LM 精炼（局部）
    print("== Levenberg-Marquardt refinement ==")
    def fun_res(p): return residuals_joint(p, data_list)
    # 把边界软约束：把变量变换到无界（这里直接用 bounds 中点截断，LM内部不处理边界）
    # 简单起见，先把 DE 结果作为初值直接喂给 least_squares，并设置 bounds（支持边界）
    res_lm = least_squares(fun_res, p_de, bounds=(np.array([d_bounds[0], A_BOUNDS[0], B_BOUNDS[0], C_BOUNDS[0], ALPHA_BOUNDS[0]]),
                                                  np.array([d_bounds[1], A_BOUNDS[1], B_BOUNDS[1], C_BOUNDS[1], ALPHA_BOUNDS[1]])),
                           method="trf", loss="linear", max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=1)
    p_opt = res_lm.x
    print("LM opt  (d, A, B, C, alpha):", [f"{v:.6g}" for v in p_opt])
    print("Final SSE:", f"{np.sum(res_lm.fun**2):.6g}")

    # 结果与可视化
    d_um, A, B, C, alpha = p_opt
    print("\n== Final parameters ==")
    print(f"d (um)   = {d_um:.3f}")
    print(f"A,B,C    = {A:.6f}, {B:.6f}, {C:.6f}")
    print(f"alpha(1/um) = {alpha:.6f}")

    # 画拟合曲线与残差
    plt.figure(figsize=(9,4))
    for wn, Rexp, theta0, label in [(wn10, R10, THETA0S[0], "10deg"), (wn15, R15, THETA0S[1], "15deg")]:
        Rpred = model_reflectance(wn, theta0, p_opt)
        plt.plot(wn, Rexp, label=f"{label} data", alpha=0.7)
        plt.plot(wn, Rpred, label=f"{label} fit")
    plt.xlabel("wavenumber sigma (cm$^{-1}$)"); plt.ylabel("reflectance")
    plt.title("Joint fit on two angles (data vs model)")
    plt.legend(); plt.tight_layout()
    fig_fit = os.path.join(BASE_DIR, "Q2_jointfit_curves.png")
    plt.savefig(fig_fit, dpi=200); plt.close()

    # 残差
    plt.figure(figsize=(9,4))
    for wn, Rexp, theta0, label in [(wn10, R10, THETA0S[0], "10deg"), (wn15, R15, THETA0S[1], "15deg")]:
        Rpred = model_reflectance(wn, theta0, p_opt)
        res = Rpred - Rexp
        plt.plot(wn, res, label=f"{label} residual")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("wavenumber sigma (cm$^{-1}$)"); plt.ylabel("residual")
    plt.title("Residuals (model - data)")
    plt.legend(); plt.tight_layout()
    fig_res = os.path.join(BASE_DIR, "Q2_jointfit_residuals.png")
    plt.savefig(fig_res, dpi=200); plt.close()

    # 10度 FFT 频谱示意（用于说明 d0 的来源）
    wn_sorted = np.sort(wn10)
    d_sigma = np.median(np.diff(wn_sorted))
    grid = np.arange(wn_sorted[0], wn_sorted[-1]+d_sigma, d_sigma)
    R_interp = np.interp(grid, wn10, df10["reflectance"].values)
    R_sg = savgol_filter(R_interp, SG_WIN, SG_POLY, mode="interp")
    bg = moving_average(R_sg, BG_WIN)
    y = R_sg - bg
    Y = np.abs(rfft(y))
    freqs = rfftfreq(len(y), d=d_sigma)
    plt.figure(figsize=(7,4))
    plt.plot(freqs, Y)
    plt.xlim(0, freqs.max())
    plt.xlabel("frequency in sigma domain (cycles per cm^-1)")
    plt.ylabel("|FFT|")
    plt.title("FFT of 10deg fringe (for d0 estimation)")
    plt.tight_layout()
    fig_fft = os.path.join(BASE_DIR, "Q2_fft_10deg.png")
    plt.savefig(fig_fft, dpi=200); plt.close()

    # 保存 CSV
    rows = [
        ["param", "value"],
        ["d_um", f"{d_um:.6f}"],
        ["A", f"{A:.9f}"],
        ["B", f"{B:.9f}"],
        ["C", f"{C:.9f}"],
        ["alpha(1/um)", f"{alpha:.9f}"],
        ["SSE", f"{np.sum(res_lm.fun**2):.9f}"],
    ]
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, header=True, encoding="utf-8-sig")

    print("\nCSV saved:", CSV_PATH)
    print("Figures saved:")
    for p in [fig_fit, fig_res, fig_fft]:
        print(" -", p)

if __name__ == "__main__":
    main()
