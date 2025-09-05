# P_B

---

# 一、整体思路（可放“方法”小节）

**目标**：联合利用两组入射角（10°、15°）的反射谱数据，基于“**两束干涉 + Cauchy 色散 + 指数衰减**”模型，**一次性**拟合出外延层厚度 $d$ 及色散/吸收参数 $(A,B,C,\alpha)$，用全谱信息最大化约束，获得稳健的 $d$。

**策略**：两阶段优化

1. **全局**：差分进化（DE）在物理边界内粗搜，避免落入局部最优；
2. **局部**：以 DE 结果为初值，用 `least_squares`（LM/trf）做精细最小二乘拟合。

**辅助**：先对 10° 光谱做 **FFT**，从条纹主频估算 $d_0$，把 $d$ 的搜索区间设为 $[0.8d_0,1.2d_0]$，显著加快收敛并稳住可辨识性。

---

# 二、物理与数学模型（代码对应）

1. **折射率（Cauchy）**

$$
n(\lambda)=A+\frac{B}{\lambda^2}+\frac{C}{\lambda^4},\quad \lambda(\mu\text{m})=\frac{10^4}{\sigma(\text{cm}^{-1})}.
$$

代码：`n_cauchy(lambda_um, A, B, C)`

2. **膜内折射角（斯涅尔）**

$$
\cos\theta_1=\sqrt{1-\left(\frac{\sin\theta_0}{n}\right)^2}.
$$

代码：`cos_theta_in(n1, theta0, n0=1.0)`

3. **未偏振菲涅尔反射（s/p 平均）**
   界面反射：`fresnel_unpolarized_R(n_in, n_out, theta_in)`

* 上表面：空气–膜 $R_1$（入射角 = $\theta_0$）
* 下表面：膜–衬底 $R_{12}$（入射角 = $\theta_1$）

> 注：衬底与外延折射率非常接近。代码里用一个**固定极小偏移** `EFF_NS_OFFSET=0.02` 构造 $n_2=n_1+0.02$，既不额外引入参数，又能产生合理的下表面反射，用于形成条纹对比度的“第二束”。

4. **两束干涉 + 指数衰减**

$$
\Delta\phi=4\pi\,n(\lambda)\,d\,\cos\theta_1 \cdot \sigma,\qquad
R_2' = R_{12}\,(1-R_1)^2\,e^{-\alpha d_{\mu m}}
$$

$$
R_{\text{model}}(\sigma)=R_1 + R_2' + 2\sqrt{R_1R_2'}\cos(\Delta\phi),
$$

其中 $d_{\mu m}$ 为厚度的微米单位，$\alpha$ 的单位是 $\mu\text{m}^{-1}$（更直观，便于约束）。
代码：`model_reflectance(wn, theta0, params)`（内部把 $d_{\mu m}$ 转换为 $d_{\text{cm}}$ 以匹配 $\sigma$ 的 cm$^{-1}$ 单位）

5. **联合残差（两角合并）**
   把 10°、15° 两条曲线的残差拼接成一个长残差向量：
   代码：`residuals_joint(p, data_list)`；SSE：`sse_of_p(p, data_list)`

---

# 三、数据流与步骤

1. **读取与预处理**

* 读取 `E:\mnt\data\附件1.xlsx`（10°）、`附件2.xlsx`（15°），两列：波数 σ（cm⁻¹）、反射率 R（若是百分比自动转 0–1）。
* **轻度平滑**（Savitzky–Golay）+ **去慢变基线**（移动平均），得到适合拟合的曲线（降低噪声、抑制光源与器件趋势项）。
  代码：`read_two_cols_xlsx`、`preprocess`

2. **FFT 估计厚度初值 $d_0$**（用 10°）

* 对去基线的条纹信号 $y(\sigma)$ 做 FFT，找主峰频率 $f_{\text{peak}}$（单位：cycles per cm⁻¹）。
* 对相位 $\cos(4\pi n d \cos\theta_1 \sigma)$ 而言，频率 $f=2 n d \cos\theta_1$，故

  $$
  d_{\text{cm}}\approx \frac{f_{\text{peak}}}{2\,n_{\text{avg}}\,\cos\theta_1},\quad d_{\mu m}=10^4 d_{\text{cm}}.
  $$

  代码里用 $n_{\text{avg}}=2.6$ 和 $\theta_0=10^\circ$ 得出 $d_0$。
  代码：`estimate_d0_via_fft`

3. **参数与边界（B2 口径）**

$$
p=[d_{\mu m},\,A,\,B,\,C,\,\alpha],\quad
\begin{cases}
d\in[0.8d_0,1.2d_0]\\
A\in[2.4,2.8]\\
B\in[-10^4,10^4]\\
C\in[-10^8,10^8]\\
\alpha\in[0,0.1]\ \ (\mu\text{m}^{-1})
\end{cases}
$$

代码：`bounds = [d_bounds, A_BOUNDS, B_BOUNDS, C_BOUNDS, ALPHA_BOUNDS]`

4. **两阶段优化**

* **差分进化（DE）**：在上述边界内最小化 SSE，得到全局较优的 $p_{\text{DE}}$。
  代码：`differential_evolution(fun_sse, bounds, ...)`
* **LM/trf 精炼**：以 $p_{\text{DE}}$ 为初值做 `least_squares(fun_res, bounds=...)`，得到最终参数 $p_{\text{opt}}$。
  代码：`least_squares(fun_res, p_de, bounds=..., method="trf")`

---

# 四、脚本中的主要变量（写到“变量表/符号表”）

* `BASE_DIR`：数据与输出目录（固定 `E:\mnt\data`）
* `FILE_10` / `FILE_15`：10° / 15° 的 Excel 文件名
* `THETA0S`：外部入射角弧度数组 `[10°, 15°]`
* `SG_WIN, SG_POLY, BG_WIN`：平滑与去基线的窗口参数
* `N_AVG_FOR_FFT`：FFT 粗估厚度时用的平均折射率（默认 2.6）
* `A_BOUNDS,B_BOUNDS,C_BOUNDS,ALPHA_BOUNDS`：Cauchy 与衰减系数的物理边界
* `EFF_NS_OFFSET`：衬底等效折射率相对外延的**微小偏移**（固定常量 0.02），用于产生合理的 $R_{12}$ 而不增加待估参数
* `model_reflectance(wn, theta0, params)`：给定参数生成模型反射率
* `residuals_joint(p, data_list)`：拼接两角残差
* `estimate_d0_via_fft(...)`：由 FFT 得厚度初值与厚度搜索区间

---

# 五、脚本输出（文件与作用）

1. **控制台输出**（ASCII 文本，便于复制到论文/补充材料）

* FFT 初值与搜索区间：`estimated d0 (um) ~ ...`
* DE 全局搜索最优点与 SSE：`DE best (d, A, B, C, alpha) ...`
* LM 精炼后最终参数：`LM opt (d, A, B, C, alpha) ...` 与最终 SSE
* 最终结果总表：

  * `d (um)`：外延层厚度**最终值**（论文核心结果）
  * `A,B,C`：该样品/波段的 Cauchy 系数（可放方法或附录）
  * `alpha (1/um)`：等效吸收系数（说明条纹对比度）

2. **CSV**：`E:\mnt\data\Q2_jointfit_results.csv`

* 两列表：参数名、数值。可直接粘贴到论文表格或做后续统计。

3. **图像**

* `Q2_jointfit_curves.png`：**两角数据 vs. 模型曲线**（最关键图，展示拟合优度与物理一致性）
* `Q2_jointfit_residuals.png`：**两角残差–波数**（应近似零均值白噪声；若有系统性趋势可在讨论中解释）
* `Q2_fft_10deg.png`：**10° 条纹 FFT 频谱**（展示 $d_0$ 的来源，作为交叉验证）

---

# 六、如何在论文里使用这些输出

* **主结论**：引用 `Q2_jointfit_results.csv` 的 `d (um)` 作为厚度结果；同时给出 10°、15° 两角全球谱联合拟合的曲线图（`curves.png`）和残差图（`residuals.png`），以“全谱一致性 + 残差无结构”为证据。
* **可重复性**：说明 `bounds` 的物理来源（A 的窄范围、B/C 的宽松量级、α 的小值上界），以及 FFT 初值如何设定，便于他人复现实验。
* **稳健性与讨论**：

  * 对比 `d (um)` 与 FFT 得到的 $d_0$ 在同一量级；
  * 说明采用两角联合拟合提升了可辨识性；
  * 如有需要，给出“固定 $A,B,C,\alpha$ 后分别对 10° / 15° 仅拟合 $d$ 的一致性复检”（可在附录）。

---

# 七、常见调参点（遇到数据差也好用）

* 条纹很弱：适当放宽 `ALPHA_BOUNDS` 上限（但不宜过大），或减小 `EFF_NS_OFFSET`；
* 收敛慢：降低 DE 的 `popsize`（如 20）、`maxiter`（如 120），并稍微放宽 `d_bounds`；
* 残差有“周期性”：提示可能存在**多光束效应/色散拟合不足**，可在问题三中引入更完整的多重反射或 Lorentz 声子项。

---

