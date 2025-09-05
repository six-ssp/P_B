"""
B题·第二问 — 结果评估
读取主脚本输出的参数文件 B2_结果以及重要参数.csv，
在固定 (A,B,C,alpha) 下评估两角（10度/15度）的拟合优度，
并做“单角仅优化 d”的一致性复检。

输出：
  - E:\B\data\B2_评估指标.csv
  - E:\B\data\B2_评估_实测对比模型_曲线图.png
  - E:\B\data\B2_评估_残差_波数图.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

# ---- Windows 控制台尽量使用 UTF-8（打印中文更稳）----
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

# ---------------- 配置（与主脚本一致） ----------------
BASE_DIR = r"E:\B\data"
FILE_10  = "附件1.xlsx"     # 10 度
FILE_15  = "附件2.xlsx"     # 15 度
PARAM_CSV = "B2_结果以及重要参数.csv"

THETA0S = [np.deg2rad(10.0), np.deg2rad(15.0)]

# 预处理参数（与主脚本保持一致）
SG_WIN  = 11
SG_POLY = 2
BG_WIN  = 151

# 与主脚本一致：用于产生非零膜-衬底反差
EFF_NS_OFFSET = 0.02

# ---------------- I/O 与预处理 ----------------
def read_two_cols_xlsx(path):
    df = pd.read_excel(path, header=None, engine="openpyxl")
    if isinstance(df.iloc[0,0], str):
        df = df.iloc[1:].copy()
    df.columns = ["wavenumber", "reflectance"]
    df["wavenumber"]  = pd.to_numeric(df["wavenumber"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")
    df = df.dropna().sort_values("wavenumber").reset_index(drop=True)
    if df["reflectance"].max() > 1.5:
        df["reflectance"] = df["reflectance"]/100.0
    return df

def moving_average(x, win):
    if win < 3: return x.copy()
    if win % 2 == 0: win += 1
    k = np.ones(win)/win
    return np.convolve(x, k, mode="same")

def preprocess(wn, R):
    """与主脚本一致：SG 平滑 + 去慢变基线；返回去基线信号 y 与平滑曲线 R_sg（评估用后者）"""
    from scipy.signal import savgol_filter
    R_sg = savgol_filter(R, SG_WIN, SG_POLY, mode="interp")
    bg   = moving_average(R_sg, BG_WIN)
    y    = R_sg - bg
    return y, R_sg

# ---------------- 光学模型（与主脚本一致） ----------------
def n_cauchy(lambda_um, A, B, C):
    return A + B/(lambda_um**2) + C/(lambda_um**4)

def cos_theta_in(n1, theta0, n0=1.0):
    s = np.sin(theta0) * n0 / n1
    s = np.clip(s, 0.0, 0.999999)
    return np.sqrt(1.0 - s*s)

def fresnel_unpolarized_R(n_in, n_out, theta_in):
    cos_in = np.cos(theta_in)
    s = np.sin(theta_in) * n_in / n_out
    s = np.clip(s, 0.0, 0.999999)
    cos_out = np.sqrt(1.0 - s*s)
    rs = (n_in*cos_in - n_out*cos_out) / (n_in*cos_in + n_out*cos_out)
    rp = (n_out*cos_in - n_in*cos_out) / (n_out*cos_in + n_in*cos_out)
    return 0.5*(rs*rs + rp*rp)

def model_reflectance(wn, theta0, params):
    """params = [d_um, A, B, C, alpha_um_inv]"""
    d_um, A, B, C, alpha = params
    d_cm = d_um * 1e-4                           # μm → cm（匹配 σ 的 cm^-1）
    lam_um = 1e4 / wn
    n1 = n_cauchy(lam_um, A, B, C)
    cos_t1 = cos_theta_in(n1, theta0, 1.0)

    n0 = 1.0
    n2 = n1 + EFF_NS_OFFSET
    R1  = fresnel_unpolarized_R(n0, n1, theta0)
    theta1 = np.arccos(cos_t1)
    R12 = fresnel_unpolarized_R(n1, n2, theta1)

    dphi = 4.0*np.pi*n1*cos_t1*wn*d_cm
    R2p  = R12 * (1.0 - R1)**2 * np.exp(-alpha * d_um)

    Rm = R1 + R2p + 2.0*np.sqrt(R1*R2p)*np.cos(dphi)
    return np.clip(Rm, 0.0, 1.0)

# ---------------- 指标与单角复检 ----------------
def rmse(y, yhat):
    r = (yhat - y)
    return float(np.sqrt(np.mean(r*r)))

def fit_d_single_angle(wn, Rexp, theta0, A, B, C, alpha, d_center_um):
    """固定 A,B,C,alpha，仅优化 d（以全局 d 为中心 ±50%）"""
    def res_d(dvar):
        p = [float(dvar[0]), A, B, C, alpha]
        return model_reflectance(wn, theta0, p) - Rexp
    lo = max(0.1, 0.5 * d_center_um)
    hi = 1.5 * d_center_um
    sol = least_squares(res_d, x0=[d_center_um],
                        bounds=([lo], [hi]),
                        method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                        max_nfev=2000)
    return float(sol.x[0])

# ---------------- 读取参数 ----------------
def load_params(path_params):
    dfp = pd.read_csv(path_params)
    # 兼容两列表头：["参数","数值"] 或 ["param","value"]，以及不同的行名写法
    key_col = dfp.columns[0]
    val_col = dfp.columns[1]
    kv = {str(k).strip(): str(v).strip() for k, v in zip(dfp[key_col], dfp[val_col])}

    # 宽松匹配键名
    def find_float(*candidates, default=None):
        for cand in candidates:
            for k, v in kv.items():
                if cand in k:
                    try:
                        return float(v)
                    except:
                        pass
        return default

    d_um  = find_float("d_um", "d（厚度", "d (um)", default=None)
    A     = find_float("A", default=None)
    B     = find_float("B", default=None)
    C     = find_float("C", default=None)
    alpha = find_float("alpha", "α", default=None)

    if None in (d_um, A, B, C, alpha):
        raise ValueError("参数CSV解析失败：未找到 d_um / A / B / C / alpha。请检查文件内容。")

    return d_um, A, B, C, alpha

# ---------------- 主流程 ----------------
def main():
    path10 = os.path.join(BASE_DIR, FILE_10)
    path15 = os.path.join(BASE_DIR, FILE_15)
    path_params = os.path.join(BASE_DIR, PARAM_CSV)
    if not (os.path.exists(path10) and os.path.exists(path15) and os.path.exists(path_params)):
        raise FileNotFoundError("缺少必要文件：附件1.xlsx、附件2.xlsx 或 B2_结果以及重要参数.csv。")

    # 读参数
    d_um, A, B, C, alpha = load_params(path_params)
    print("== 读取拟合参数 ==")
    print(f"d = {d_um:.6f} μm,  A = {A:.9f},  B = {B:.9f},  C = {C:.9f},  α = {alpha:.9f}  (1/μm)")

    # 读光谱并预处理
    df10 = read_two_cols_xlsx(path10)
    df15 = read_two_cols_xlsx(path15)
    wn10 = df10["wavenumber"].values.astype(float)
    wn15 = df15["wavenumber"].values.astype(float)
    _, R10 = preprocess(wn10, df10["reflectance"].values)
    _, R15 = preprocess(wn15, df15["reflectance"].values)

    # 用全局参数生成模型曲线
    p_opt = [d_um, A, B, C, alpha]
    Rpred10 = model_reflectance(wn10, THETA0S[0], p_opt)
    Rpred15 = model_reflectance(wn15, THETA0S[1], p_opt)

    # 评估指标
    sse10 = float(np.sum((Rpred10 - R10)**2))
    sse15 = float(np.sum((Rpred15 - R15)**2))
    rmse10_ = rmse(R10, Rpred10)
    rmse15_ = rmse(R15, Rpred15)
    rmse_total = float(np.sqrt((sse10 + sse15) / (len(wn10) + len(wn15))))
    res10_mu = float(np.mean(Rpred10 - R10))
    res10_sd = float(np.std (Rpred10 - R10, ddof=1))
    res15_mu = float(np.mean(Rpred15 - R15))
    res15_sd = float(np.std (Rpred15 - R15, ddof=1))

    # 单角一致性复检（仅优化 d）
    d10_only = fit_d_single_angle(wn10, R10, THETA0S[0], A, B, C, alpha, d_um)
    d15_only = fit_d_single_angle(wn15, R15, THETA0S[1], A, B, C, alpha, d_um)
    d_only_avg   = 0.5*(d10_only + d15_only)
    rel_diff_pct = float(abs(d10_only - d15_only) / d_only_avg * 100.0)

    # 打印摘要
    print("\n== 结果评估（摘要）==")
    print(f"整体 RMSE = {rmse_total:.6g}  |  整体 SSE = {(sse10+sse15):.6g}")
    print(f"10度 RMSE = {rmse10_:.6g}     |  15度 RMSE = {rmse15_:.6g}")
    print(f"10度残差：均值 = {res10_mu:.3e}，标准差 = {res10_sd:.3e}")
    print(f"15度残差：均值 = {res15_mu:.3e}，标准差 = {res15_sd:.3e}")
    print(f"单角复检：d_10 = {d10_only:.4f} μm，d_15 = {d15_only:.4f} μm")
    print(f"两角一致性（仅优化 d）= {rel_diff_pct:.3f} %")

    # 保存评估CSV
    eval_rows = [
        ["指标", "数值"],
        ["d_全局(μm)", f"{d_um:.9g}"],
        ["A", f"{A:.9g}"],
        ["B", f"{B:.9g}"],
        ["C", f"{C:.9g}"],
        ["alpha(1/um)", f"{alpha:.9g}"],
        ["RMSE_总体", f"{rmse_total:.9g}"],
        ["SSE_总体",  f"{(sse10+sse15):.9g}"],
        ["RMSE_10度", f"{rmse10_:.9g}"],
        ["RMSE_15度", f"{rmse15_:.9g}"],
        ["残差均值_10度", f"{res10_mu:.9g}"],
        ["残差标准差_10度",  f"{res10_sd:.9g}"],
        ["残差均值_15度", f"{res15_mu:.9g}"],
        ["残差标准差_15度",  f"{res15_sd:.9g}"],
        ["d_10度_仅优化d(μm)", f"{d10_only:.9g}"],
        ["d_15度_仅优化d(μm)", f"{d15_only:.9g}"],
        ["两角一致性_仅优化d(%)", f"{rel_diff_pct:.9g}"],
    ]
    eval_csv = os.path.join(BASE_DIR, "B2_评估指标.csv")
    pd.DataFrame(eval_rows).to_csv(eval_csv, index=False, header=True, encoding="utf-8-sig")
    print("\n评估指标 CSV 已保存：", eval_csv)

    # 图像：数据 vs 模型
    plt.figure(figsize=(9,4))
    plt.plot(wn10, R10, label="10度 实测", alpha=0.7)
    plt.plot(wn10, Rpred10, label="10度 模型")
    plt.plot(wn15, R15, label="15度 实测", alpha=0.7)
    plt.plot(wn15, Rpred15, label="15度 模型")
    plt.xlabel("波数 σ (cm$^{-1}$)")
    plt.ylabel("反射率")
    plt.title("评估：两角实测 vs 模型")
    plt.legend(); plt.tight_layout()
    fig1 = os.path.join(BASE_DIR, "B2_评估_实测对比模型_曲线图.png")
    plt.savefig(fig1, dpi=200); plt.close()

    # 图像：残差
    plt.figure(figsize=(9,4))
    plt.plot(wn10, Rpred10 - R10, label="10度 残差")
    plt.plot(wn15, Rpred15 - R15, label="15度 残差")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("波数 σ (cm$^{-1}$)")
    plt.ylabel("残差")
    plt.title("评估：残差（模型 - 实测）")
    plt.legend(); plt.tight_layout()
    fig2 = os.path.join(BASE_DIR, "B2_评估_残差_波数图.png")
    plt.savefig(fig2, dpi=200); plt.close()

    print("图像已保存：")
    print(" -", fig1)
    print(" -", fig2)

if __name__ == "__main__":
    main()
