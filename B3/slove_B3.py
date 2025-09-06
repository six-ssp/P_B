# -*- coding: utf-8 -*-
"""
B3  —  多次反射与透射（Airy 多光束）联合拟合
口径与B2一致：柯西 n(λ)=A+B/λ^2+C/λ^4，常数吸收 alpha。
不读取B2输出；仅用附件3(10°)、附件4(15°)原始xlsx。
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution, least_squares
from numpy.fft import rfft, rfftfreq

# -------- 显示设置 --------
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","DejaVu Sans","Noto Sans CJK SC","Arial Unicode MS"]

# -------- 数据路径 --------
BASE_DIR = r"E:\B\data"
FILE_10 = "附件3.xlsx"   # 10°
FILE_15 = "附件4.xlsx"   # 15°

# -------- 预处理参数 --------
SG_WIN, SG_POLY = 11, 2
BG_WIN = 151

# -------- 参数边界（与B2风格一致） --------
A_BOUNDS     = (2.4, 2.8)
B_BOUNDS     = (-1e4, 1e4)
C_BOUNDS     = (-1e8, 1e8)
ALPHA_BOUNDS = (0.0, 0.1)       # μm^-1
EFF_NS_OFFSET = 0.02            # 基底等效折射率对外延的微偏移

# ================= 工具函数 =================
def read_two_cols_xlsx(path):
    df = pd.read_excel(path, header=None, engine="openpyxl")
    if isinstance(df.iloc[0,0], str):
        df = df.iloc[1:].copy()
    df.columns = ["wavenumber","reflectance"]
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
    R_sg = savgol_filter(R, SG_WIN, SG_POLY, mode="interp")
    bg   = moving_average(R_sg, BG_WIN)
    y    = R_sg - bg
    return y, R_sg

# ---- 柯西 + 吸收 ----
def n_cauchy(lambda_um, A, B, C):
    return A + B/(lambda_um**2) + C/(lambda_um**4)

def kappa_from_alpha(lambda_um, alpha):
    return alpha * lambda_um / (4.0*np.pi)  # α(μm^-1) -> κ

def snell_cos_theta(n_complex, theta0, n0=1.0):
    n_real = np.real(n_complex)
    s = np.sin(theta0)*n0/np.maximum(n_real,1e-6)
    s = np.clip(s, 0.0, 0.999999)
    return np.sqrt(1.0 - s*s)

# ---- Airy(矩阵法) ----
def characteristic_matrix_single_layer(n0, n1c, ns, theta0, d_um, lambda_um, pol="s"):
    cos0 = np.cos(theta0)
    cos1 = snell_cos_theta(n1c, theta0, n0=n0)
    coss = snell_cos_theta(ns,   theta0, n0=n0)
    if pol=="s":
        q0, q1, qs = n0*cos0, n1c*cos1, ns*coss
    else:
        q0, q1, qs = n0/cos0, n1c/cos1, ns/coss
    delta = 2.0*np.pi * n1c * d_um * 1e-4 * cos1 / lambda_um
    M11 = np.cos(delta); M22 = M11
    js  = 1j*np.sin(delta)
    M12 = js / q1; M21 = js * q1
    Y = (M11*qs + M12) / (M21*qs + M22)
    r = (q0 - Y) / (q0 + Y)
    return r

def R_airy(wn, theta0, p):
    d_um,A,B,C,alpha = p
    lam = 1.0e4/wn
    n1  = n_cauchy(lam,A,B,C)
    k1  = kappa_from_alpha(lam,alpha)
    n1c = n1 - 1j*k1
    ns  = n1 + EFF_NS_OFFSET     # 基底取实数微偏移
    r_s = characteristic_matrix_single_layer(1.0, n1c, ns, theta0, d_um, lam, "s")
    r_p = characteristic_matrix_single_layer(1.0, n1c, ns, theta0, d_um, lam, "p")
    return np.clip(0.5*(np.abs(r_s)**2 + np.abs(r_p)**2), 0.0, 1.0)

# ---- 两束近似（仅用于粗搜/对照） ----
def R_twobeam(wn, theta0, p):
    d_um,A,B,C,alpha = p
    d_cm = d_um*1e-4
    lam = 1.0e4/wn
    n1  = n_cauchy(lam,A,B,C)
    n2  = n1 + EFF_NS_OFFSET
    cos1 = snell_cos_theta(n1, theta0, 1.0)
    def fresnel_unpol(n_in,n_out,theta_in):
        ci = np.cos(theta_in)
        s  = np.sin(theta_in)*n_in/n_out
        s  = np.clip(s,0,0.999999); co = np.sqrt(1.0-s*s)
        rs = (n_in*ci - n_out*co)/(n_in*ci + n_out*co)
        rp = (n_out*ci - n_in*co)/(n_out*ci + n_in*co)
        return 0.5*(rs*rs + rp*rp)
    R1  = fresnel_unpol(1.0, n1, theta0)
    R12 = fresnel_unpol(n1, n2, np.arccos(cos1))
    dphi = 4.0*np.pi*n1*cos1*wn*d_cm
    R2p  = R12*(1.0-R1)**2*np.exp(-alpha*d_um)
    R    = R1 + R2p + 2.0*np.sqrt(R1*R2p)*np.cos(dphi)
    return np.clip(R,0.0,1.0)

# ---- 残差/目标 ----
def residuals_joint(p, packs, airy=True):
    res=[]
    for wn,Rexp,t in packs:
        Rm = R_airy(wn,t,p) if airy else R_twobeam(wn,t,p)
        res.append(Rm-Rexp)
    return np.concatenate(res)

def sse(p, packs, airy=True):
    r = residuals_joint(p, packs, airy)
    return float(np.dot(r,r))

# ---- FFT 厚度初值 ----
def estimate_d0_fft(wn, R, n_avg=2.6, theta0=np.deg2rad(10.0)):
    wn = np.sort(wn)
    dσ  = np.median(np.diff(wn))
    grid = np.arange(wn[0], wn[-1]+dσ, dσ)
    Rg = np.interp(grid, wn, R)
    Rg = savgol_filter(Rg, SG_WIN, SG_POLY, mode="interp")
    bg = moving_average(Rg, BG_WIN)
    y  = Rg - bg
    Y  = np.abs(rfft(y)); f = rfftfreq(len(y), d=dσ)
    m  = f>0; fp = f[m][np.argmax(Y[m])] if np.any(m) else 0.02
    cos1 = snell_cos_theta(n_avg, theta0, 1.0)
    d_cm = fp/(2.0*n_avg*cos1)
    return float(np.clip(d_cm*1e4, 0.1, 200.0))

# ================= 主程序 =================
def main():
    p10 = os.path.join(BASE_DIR, FILE_10)
    p15 = os.path.join(BASE_DIR, FILE_15)
    if not (os.path.exists(p10) and os.path.exists(p15)):
        raise FileNotFoundError("请确认 E:\\B\\data\\附件3.xlsx 与 附件4.xlsx 存在")

    df10, df15 = read_two_cols_xlsx(p10), read_two_cols_xlsx(p15)
    wn10, R10r = df10["wavenumber"].to_numpy(float), df10["reflectance"].to_numpy(float)
    wn15, R15r = df15["wavenumber"].to_numpy(float), df15["reflectance"].to_numpy(float)
    y10, R10 = preprocess(wn10, R10r)
    y15, R15 = preprocess(wn15, R15r)

    TH10, TH15 = np.deg2rad(10.0), np.deg2rad(15.0)
    packs = [(wn10,R10,TH10),(wn15,R15,TH15)]

    d0 = estimate_d0_fft(wn10, R10r, n_avg=2.6, theta0=TH10)
    d_bounds = (0.8*d0, 1.2*d0)
    print(f"[FFT] d0≈{d0:.2f} μm,  搜索区间：{d_bounds[0]:.2f}~{d_bounds[1]:.2f} μm")

    bounds = [d_bounds, A_BOUNDS, B_BOUNDS, C_BOUNDS, ALPHA_BOUNDS]

    print("\n[阶段1] 两束近似·差分进化（粗搜）")
    de1 = differential_evolution(lambda p: sse(p, packs, airy=False),
                                 bounds=bounds, strategy="best1bin",
                                 maxiter=120, popsize=25, tol=1e-6, seed=42, polish=False)
    p0 = de1.x
    print("  初值(两束DE)：", [f"{v:.6g}" for v in p0], "  SSE:", f"{de1.fun:.6g}")

    print("\n[阶段2] Airy·差分进化（微调）")
    de2 = differential_evolution(lambda p: sse(p, packs, airy=True),
                                 bounds=bounds, strategy="best1bin",
                                 maxiter=80, popsize=15, tol=1e-6, seed=43, polish=False)
    p1 = de2.x
    print("  初值(Airy DE)：", [f"{v:.6g}" for v in p1], "  SSE:", f"{de2.fun:.6g}")

    print("\n[阶段3] Airy·最小二乘（最终精炼）")
    lb = np.array([d_bounds[0], A_BOUNDS[0], B_BOUNDS[0], C_BOUNDS[0], ALPHA_BOUNDS[0]])
    ub = np.array([d_bounds[1], A_BOUNDS[1], B_BOUNDS[1], C_BOUNDS[1], ALPHA_BOUNDS[1]])
    ls = least_squares(lambda p: residuals_joint(p, packs, airy=True), p1,
                       bounds=(lb,ub), method="trf", loss="linear",
                       max_nfev=6000, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=1)
    p = ls.x
    print("\n[结果] p*=[d_um, A, B, C, alpha] =", [f"{v:.9g}" for v in p])
    print("      SSE =", f"{np.sum(ls.fun**2):.6g}")

    # ---------- 出图 ----------
    def plot_fit(wn, Rexp, th, tag, out):
        Rt = R_twobeam(wn, th, p)
        Ra = R_airy(wn, th, p)
        rm_t = np.sqrt(np.mean((Rt-Rexp)**2))
        rm_a = np.sqrt(np.mean((Ra-Rexp)**2))
        plt.figure(figsize=(12,5))
        plt.plot(wn, Rexp, ".", ms=2.5, alpha=0.9, label="实测")
        plt.plot(wn, Rt, label=f"两束近似  RMSE={rm_t:.4f}")
        plt.plot(wn, Ra, label=f"Airy 多光束 RMSE={rm_a:.4f}")
        plt.xlabel("波数 σ (cm$^{-1}$)"); plt.ylabel("反射率")
        plt.title(f"拟合：实测 vs 模型 —— {tag}")
        plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()

    def plot_res(wn, Rexp, th, tag, out):
        Rt = R_twobeam(wn, th, p); Ra = R_airy(wn, th, p)
        rm_t = np.sqrt(np.mean((Rt-Rexp)**2)); rm_a = np.sqrt(np.mean((Ra-Rexp)**2))
        plt.figure(figsize=(12,5))
        plt.plot(wn, Rt-Rexp, label=f"两束残差  RMSE={rm_t:.4f}")
        plt.plot(wn, Ra-Rexp, label=f"Airy残差  RMSE={rm_a:.4f}")
        plt.axhline(0, color="k", lw=0.8)
        plt.xlabel("波数 σ (cm$^{-1}$)"); plt.ylabel("残差")
        plt.title(f"残差对比 —— {tag}")
        plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()

    def plot_fft_res(wn, Rexp, th, tag, out):
        dσ  = np.median(np.diff(np.sort(wn)))
        grid = np.arange(wn[0], wn[-1]+dσ, dσ)
        Ra = R_airy(grid, th, p); Rgi = np.interp(grid, wn, Rexp)
        res = Ra - Rgi
        Y = np.abs(rfft(res)); f = rfftfreq(len(res), d=dσ)
        plt.figure(figsize=(12,5)); plt.semilogy(f, Y, label="Airy残差谱")
        plt.xlabel("频率（σ域，cycles/cm$^{-1}$）"); plt.ylabel("|FFT(残差)|（对数）")
        plt.title(f"残差频谱 —— {tag}")
        plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()

    paths = {
        "fit3": os.path.join(BASE_DIR, "B3_拟合_数据对模型_附件3.png"),
        "fit4": os.path.join(BASE_DIR, "B3_拟合_数据对模型_附件4.png"),
        "res3": os.path.join(BASE_DIR, "B3_残差_附件3.png"),
        "res4": os.path.join(BASE_DIR, "B3_残差_附件4.png"),
        "fft3": os.path.join(BASE_DIR, "B3_残差谱FFT_附件3.png"),
        "fft4": os.path.join(BASE_DIR, "B3_残差谱FFT_附件4.png"),
    }
    plot_fit(wn10, R10, TH10, "附件3", paths["fit3"])
    plot_fit(wn15, R15, TH15, "附件4", paths["fit4"])
    plot_res(wn10, R10, TH10, "附件3", paths["res3"])
    plot_res(wn15, R15, TH15, "附件4", paths["res4"])
    plot_fft_res(wn10, R10, TH10, "附件3", paths["fft3"])
    plot_fft_res(wn15, R15, TH15, "附件4", paths["fft4"])

    # ---------- 导出参数 ----------
    csv_path = os.path.join(BASE_DIR, "B3_拟合参数与误差.csv")
    d_um,A,B,C,alpha = p
    pd.DataFrame(
        [["d_um",f"{d_um:.9f}"],["A",f"{A:.9f}"],["B",f"{B:.9f}"],["C",f"{C:.9f}"],
         ["alpha(1/um)",f"{alpha:.9f}"],["SSE(Airy)",f"{np.sum(ls.fun**2):.9f}"]],
        columns=["参数","取值"]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n导出：")
    print("  参数CSV ->", csv_path)
    for k,v in paths.items(): print(" ", v)

if __name__ == "__main__":
    main()
