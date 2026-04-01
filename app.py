"""
Streamlit Web Application
Facade Insulation Optimization using Neural Network
Topic: OPTIMIZING THE INSULATION LAYER FOR ENERGY EFFICIENCY IN BUILDING FACADES
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Facade Insulation Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE — чтобы результаты не исчезали
# ─────────────────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None      # результат симуляции
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None      # результат оптимизации
if "sens_result" not in st.session_state:
    st.session_state.sens_result = None     # результат sensitivity analysis

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; }
    .block-container { padding: 2rem 3rem; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-box {
        background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    .metric-unit { font-size: 0.8rem; color: #8b949e; margin-left: 4px; }

    .result-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid #21a0a0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .result-value {
        font-size: 2.5rem;
        font-family: 'Space Mono', monospace;
        color: #00d4aa;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-green  { background: #1a4731; color: #3fb950; border: 1px solid #3fb950; }
    .badge-blue   { background: #0c2d6b; color: #58a6ff; border: 1px solid #58a6ff; }
    .badge-orange { background: #3d2000; color: #f0883e; border: 1px solid #f0883e; }

    .stSlider > div > div > div > div { background: #21a0a0; }
    .stButton > button {
        background: linear-gradient(135deg, #21a0a0, #0d7377);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #26c6c6, #148f94);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(33,160,160,0.4);
    }
    .info-block {
        background: #161b22;
        border-left: 3px solid #21a0a0;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PHYSICS ENGINE
# ─────────────────────────────────────────────
L       = 0.3
H       = 3.0
k_brick = 0.75
k_insul = 0.035
delta   = 0.01
margin  = delta * L
tol     = 1e-6

def h_out(y, V_inf):
    h0, h1, V0, y0, lam = 5.82, 3.96, 1.0, 65.33, 0.32
    return h0 + h1 * (V_inf / V0) * (y / y0) ** lam

def solar_radiation_fn(y, q0):
    return q0 * np.tan((np.pi / 4.0) * (y / H))

def harmonic(a, b):
    return 2.0 * a * b / (a + b + 1e-30)

def face_conductivities(k):
    ke = np.zeros_like(k); kw = np.zeros_like(k)
    kn = np.zeros_like(k); ks = np.zeros_like(k)
    ke[:, :-1] = harmonic(k[:, :-1], k[:, 1:])
    kw[:, 1:]  = ke[:, :-1]
    kn[:-1, :] = harmonic(k[:-1, :], k[1:, :])
    ks[1:,  :] = kn[:-1, :]
    return ke, kw, kn, ks

def Gamma(y, p0, p1):
    eta = y / H
    return 0.5 * L + p0 * (eta * (eta - p1) * (eta - 1.0) - (2*p1 - 1.0)/12.0)

def admissible_p0_bounds(p1, ys_high):
    eta = ys_high / H
    G   = eta*(eta - p1)*(eta - 1.0) - (2*p1 - 1.0)/12.0
    lower, upper = [], []
    for g in G:
        if abs(g) < 1e-14: continue
        p_low  = (margin - 0.5*L) / g
        p_high = (L - margin - 0.5*L) / g
        lower.append(min(p_low, p_high))
        upper.append(max(p_low, p_high))
    if not lower: return None, None
    return max(lower), min(upper)

def build_k_map(interface, x_grid, Ny, Nx):
    k_map = np.ones((Ny, Nx)) * k_brick
    for j in range(Ny):
        idx = np.searchsorted(x_grid, interface[j])
        if idx < Nx:
            k_map[j, idx:] = k_insul
    return k_map

def solve_temperature(k_map, x_grid, y_arr, T_left_inf, T_right_inf, V_inf, q0):
    Ny, Nx = k_map.shape
    dx = x_grid[1] - x_grid[0]
    dy = y_arr[1]  - y_arr[0]
    X, _ = np.meshgrid(x_grid, y_arr)
    T    = T_left_inf + (T_right_inf - T_left_inf) * (X / L)
    ke, kw, kn, ks = face_conductivities(k_map)
    ax    = 1.0 / dx**2
    ay    = 1.0 / dy**2
    denom = (ke + kw) * ax + (kn + ks) * ay
    hL_arr = h_out(y_arr, V_inf)
    qL_arr = solar_radiation_fn(y_arr, q0)
    for _ in range(500):
        T_old = T.copy()
        T[0, :]  = T[1, :]
        T[-1, :] = T[-2, :]
        kL = k_map[:, 0]
        T[:, 0]  = ((kL/dx)*T[:, 1]  + hL_arr*T_left_inf  + qL_arr) / (kL/dx + hL_arr)
        kR = k_map[:, -1]
        T[:, -1] = ((kR/dx)*T[:, -2] + 7.0*T_right_inf) / (kR/dx + 7.0)
        rhs = (
            (ke*np.roll(T, -1, axis=1) + kw*np.roll(T,  1, axis=1)) * ax +
            (kn*np.roll(T, -1, axis=0) + ks*np.roll(T,  1, axis=0)) * ay
        )
        T = rhs / denom
        if np.linalg.norm(T - T_old, ord=np.inf) < tol:
            break
    return T

def Q_in_total(T, y_arr, T_right_inf):
    q = 7.0 * (T[:, -1] - T_right_inf)
    return np.trapezoid(q, y_arr) if hasattr(np, 'trapezoid') else np.trapz(q, y_arr)


# ─────────────────────────────────────────────
# NEURAL NETWORK LOADER
# ─────────────────────────────────────────────
def load_model():
    try:
        import tensorflow as tf
        import joblib
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        model = tf.keras.models.load_model(os.path.join(base, "facade_mlp_model.h5"))
        scaler = joblib.load(os.path.join(base, "facade_scaler.pkl"))
        return model, scaler, True
    except Exception as e:
        st.sidebar.error(f"Model error: {e}")
        return None, None, False

model, scaler, model_loaded = load_model()

def predict_nn(p0, p1, T_out_C, T_in_C, V_inf_val, q0_val):
    x = np.array([[p0, p1, T_out_C, T_in_C, V_inf_val, q0_val]])
    return float(model.predict(scaler.transform(x), verbose=0).flatten()[0])

def run_physics(p0, p1, T_out_C, T_in_C, V_inf_val, q0_val, Nx=61, Ny=121):
    T_left_inf  = T_out_C + 273.15
    T_right_inf = T_in_C  + 273.15
    ys_high = np.linspace(0, H, 4001)
    p0_min, p0_max = admissible_p0_bounds(p1, ys_high)
    if p0_min is None or not (p0_min <= p0 <= p0_max):
        return None
    x_grid = np.linspace(0, L, Nx)
    y_arr  = np.linspace(0, H, Ny)
    x_raw  = np.clip(Gamma(ys_high, p0, p1), margin, L - margin)
    spline    = UnivariateSpline(ys_high, x_raw, s=1e-8)
    interface = np.clip(gaussian_filter1d(spline(y_arr), sigma=1.0), margin, L - margin)
    k_map = build_k_map(interface, x_grid, Ny, Nx)
    T     = solve_temperature(k_map, x_grid, y_arr, T_left_inf, T_right_inf, V_inf_val, q0_val)
    Q     = Q_in_total(T, y_arr, T_right_inf)
    return {"T": T, "k_map": k_map, "Q": Q,
            "interface": interface, "x_grid": x_grid, "y_arr": y_arr,
            "T_right_inf": T_right_inf}


# ─────────────────────────────────────────────
# HELPER — отрисовка результатов симуляции
# ─────────────────────────────────────────────
def render_simulation(res, Q_nn=None):
    col_r, col_v = st.columns([1, 2])

    with col_r:
        st.markdown("### 📊 Results")

        if Q_nn is not None:
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;letter-spacing:.1em;">
                    🤖 Neural Network Prediction
                </div>
                <div class="result-value">{Q_nn:.2f}</div>
                <div style="color:#8b949e;font-size:0.85rem;">W/m — Total Heat Flux Q<sub>in</sub></div>
            </div>""", unsafe_allow_html=True)

        if res is not None:
            st.markdown(f"""
            <div class="result-card" style="border-color:#3fb950;">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;letter-spacing:.1em;">
                    🔬 Physics Solver
                </div>
                <div class="result-value" style="color:#3fb950;">{res['Q']:.2f}</div>
                <div style="color:#8b949e;font-size:0.85rem;">W/m — Total Heat Flux Q<sub>in</sub></div>
            </div>""", unsafe_allow_html=True)

            if Q_nn is not None:
                err = abs(Q_nn - res["Q"]) / abs(res["Q"]) * 100
                st.metric("NN vs Physics Error", f"{err:.1f}%")

        st.markdown("### 💡 Interpretation")
        ref_Q = Q_nn if Q_nn is not None else (res["Q"] if res else None)
        if ref_Q is not None:
            if ref_Q > -20:
                st.success("🌿 Low heat loss — excellent insulation!")
            elif ref_Q > -40:
                st.warning("⚡ Moderate heat loss — typical conditions.")
            else:
                st.error("🔥 High heat loss — consider thicker insulation.")

    with col_v:
        st.markdown("### 🗺️ Visualization")
        if res is not None:
            T_C       = res["T"] - 273.15
            interface = res["interface"]
            x_grid    = res["x_grid"]
            y_arr     = res["y_arr"]

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor("#0d1117")

            # Temperature field
            ax1 = axes[0]
            ax1.set_facecolor("#0d1117")
            im = ax1.contourf(x_grid*100, y_arr, T_C, levels=30, cmap="RdYlBu_r")
            ax1.plot(interface*100, y_arr, "w--", linewidth=2, label="Insulation boundary")
            plt.colorbar(im, ax=ax1, label="Temperature (°C)")
            ax1.set_xlabel("Wall depth (cm)", color="#8b949e")
            ax1.set_ylabel("Height (m)",      color="#8b949e")
            ax1.set_title("Temperature Field", color="#c9d1d9", pad=10)
            ax1.tick_params(colors="#8b949e")
            ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
            for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")

            # Insulation profile
            ax2 = axes[1]
            ax2.set_facecolor("#0d1117")
            brick_p = mpatches.Patch(color="#8b6914", alpha=0.7, label="Brick")
            insul_p = mpatches.Patch(color="#21a0a0", alpha=0.7, label="Insulation")
            for j, y_val in enumerate(y_arr):
                xi = interface[j] * 100
                h  = H / len(y_arr) + 0.001
                ax2.barh(y_val, xi,         height=h, color="#8b6914", alpha=0.6, align="center")
                ax2.barh(y_val, L*100 - xi, height=h, color="#21a0a0", alpha=0.6,
                         align="center", left=xi)
            ax2.plot(interface*100, y_arr, color="#00d4aa", linewidth=2.5, label="Interface")
            ax2.set_xlim(0, L*100)
            ax2.set_xlabel("Wall depth (cm)", color="#8b949e")
            ax2.set_ylabel("Height (m)",      color="#8b949e")
            ax2.set_title("Insulation Profile", color="#c9d1d9", pad=10)
            ax2.tick_params(colors="#8b949e")
            ax2.legend(handles=[brick_p, insul_p], facecolor="#161b22",
                       labelcolor="white", fontsize=8)
            for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="color:#00d4aa;font-size:2rem;margin-bottom:0;">🏗️ Facade Insulation Optimizer</h1>
<p style="color:#8b949e;font-size:0.95rem;margin-top:0.3rem;margin-bottom:1.5rem;">
Neural Network · Heat Transfer Simulation · Energy Efficiency
</p>
""", unsafe_allow_html=True)

if model_loaded:
    st.markdown('<span class="badge badge-green">✓ Neural Network Loaded</span> '
                '<span class="badge badge-blue">✓ Physics Solver Active</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-orange">⚠ NN not found — Physics Solver only</span>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("### 🌡️ Climate Conditions")
    T_out    = st.slider("Outdoor Temperature (°C)", -30,  15,  -10, 1)
    T_in     = st.slider("Indoor Temperature (°C)",   15,  30,   22, 1)
    V_wind   = st.slider("Wind Speed (m/s)",          0.5, 10.0, 3.0, 0.5)
    q0_solar = st.slider("Solar Radiation Coefficient", 50, 400, 200, 10)

    st.markdown("### 📐 Insulation Geometry")
    st.markdown('<div class="info-block">p0 and p1 define the shape of the '
                'insulation-brick boundary along the wall height.</div>',
                unsafe_allow_html=True)
    p1_val = st.slider("p1 — shape curvature", 0.0, 1.0, 0.37, 0.01)

    ys_high = np.linspace(0, H, 4001)
    p0_min_b, p0_max_b = admissible_p0_bounds(p1_val, ys_high)
    if p0_min_b is None:
        st.error("Invalid p1 — no admissible p0 range")
        p0_val = 0.0
    else:
        p0_val = st.slider("p0 — shape amplitude",
                           float(round(p0_min_b, 3)),
                           float(round(p0_max_b, 3)),
                           float(round((p0_min_b + p0_max_b) / 2, 3)), 0.001)

    st.markdown("### 🔧 Computation")
    use_physics = st.checkbox("Run Physics Solver (slower)", value=True)
    run_btn     = st.button("▶  RUN SIMULATION")

# ─────────────────────────────────────────────
# QUICK METRICS
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">ΔT</div>'
                f'<div class="metric-value" style="color:#58a6ff">{T_in - T_out}'
                f'<span class="metric-unit">°C</span></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Wind Speed</div>'
                f'<div class="metric-value" style="color:#f0883e">{V_wind}'
                f'<span class="metric-unit">m/s</span></div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Solar Radiation</div>'
                f'<div class="metric-value" style="color:#d2a8ff">{q0_solar}'
                f'<span class="metric-unit">q₀</span></div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Geometry (p0, p1)</div>'
                f'<div class="metric-value" style="color:#00d4aa;font-size:1.2rem;">'
                f'{p0_val:.3f}, {p1_val:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────
if run_btn:
    Q_nn = None
    if model_loaded:
        with st.spinner("Neural Network predicting..."):
            Q_nn = predict_nn(p0_val, p1_val, T_out, T_in, V_wind, q0_solar)

    phys = None
    if use_physics:
        with st.spinner("Running physics simulation..."):
            phys = run_physics(p0_val, p1_val, T_out, T_in, V_wind, q0_solar)
        if phys is None:
            st.error("Physics solver: invalid geometry parameters.")

    # Сохраняем в session_state — результат не исчезнет при нажатии других кнопок
    st.session_state.sim_result = {"Q_nn": Q_nn, "phys": phys}

# Всегда показываем последний результат симуляции
if st.session_state.sim_result is not None:
    r = st.session_state.sim_result
    render_simulation(r["phys"], r["Q_nn"])

# ─────────────────────────────────────────────
# FIND OPTIMAL GEOMETRY
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Find Optimal Geometry")
st.markdown("*Automatically find the best p0/p1 to minimize heat loss for current climate conditions.*")

if model_loaded:
    if st.button("🔍  FIND OPTIMAL GEOMETRY"):
        with st.spinner("Searching optimal geometry... (50×50 grid)"):
            ys_h   = np.linspace(0, H, 4001)
            best_Q = -np.inf
            best_p0 = None
            best_p1 = None
            # Собираем все комбинации сразу
            all_p0, all_p1 = [], []
            for p1 in np.linspace(0.0, 1.0, 50):
              p0_lo, p0_hi = admissible_p0_bounds(p1, ys_h)
              if p0_lo is None or p0_lo >= p0_hi: continue
              for p0 in np.linspace(p0_lo, p0_hi, 50):
                all_p0.append(p0)
                all_p1.append(p1)

            # Один батч-запрос вместо 2500 отдельных
            X_batch = np.array([[p0, p1, T_out, T_in, V_wind, q0_solar]
                for p0, p1 in zip(all_p0, all_p1)])
            X_scaled = scaler.transform(X_batch)
            preds = model.predict(X_scaled, verbose=0).flatten()
            
            # Находим лучший
            best_idx = np.argmax(preds)
            best_Q   = float(preds[best_idx])
            best_p0  = all_p0[best_idx]
            best_p1  = all_p1[best_idx]
        # Сохраняем результат
        st.session_state.opt_result = {
            "best_p0": best_p0, "best_p1": best_p1, "best_Q": best_Q
        }

    # Всегда показываем последний результат оптимизации
    if st.session_state.opt_result is not None:
        opt = st.session_state.opt_result
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;">🏆 Optimal p0</div>
                <div class="result-value">{opt['best_p0']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;">🏆 Optimal p1</div>
                <div class="result-value">{opt['best_p1']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;">✅ Best Q_in (NN)</div>
                <div class="result-value">{opt['best_Q']:.2f}</div>
                <div style="color:#8b949e;font-size:0.8rem;">W/m</div>
            </div>""", unsafe_allow_html=True)

        current_Q   = predict_nn(p0_val, p1_val, T_out, T_in, V_wind, q0_solar)
        improvement = abs(opt["best_Q"] - current_Q)
        st.success(f"✅ Optimal geometry found!  Improvement: **{improvement:.2f} W/m** over current settings.")
        st.info(f"💡 Set  p0 = {opt['best_p0']:.4f}  and  p1 = {opt['best_p1']:.4f}  in the sidebar to visualize the optimal profile.")
else:
    st.info("Load the neural network model to use optimization.")

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Sensitivity Analysis")
st.markdown("*See how each input parameter affects the predicted heat flux Q_in.*")

if model_loaded:
    if st.button("▶  Run Sensitivity Analysis"):
        with st.spinner("Computing sensitivity..."):
            baseline = dict(p0=p0_val, p1=p1_val,
                            T_out_C=T_out, T_in_C=T_in,
                            V_inf=V_wind, q0=q0_solar)
            ys_h = np.linspace(0, H, 4001)
            p0_lo_b, p0_hi_b = admissible_p0_bounds(p1_val, ys_h)

            ranges = {
                "p0":      np.linspace(p0_lo_b, p0_hi_b, 40),
                "p1":      np.linspace(0.0,  1.0,   40),
                "T_out_C": np.linspace(-25,   5,    40),
                "T_in_C":  np.linspace(18,   26,    40),
                "V_inf":   np.linspace(0.5,   8.0,  40),
                "q0":      np.linspace(50,  400,    40),
            }
            labels = {
                "p0":      "Shape Amplitude p₀",
                "p1":      "Shape Curvature p₁",
                "T_out_C": "Outdoor Temp (°C)",
                "T_in_C":  "Indoor Temp (°C)",
                "V_inf":   "Wind Speed (m/s)",
                "q0":      "Solar Radiation q₀",
            }
            colors = ["#58a6ff","#3fb950","#f0883e","#d2a8ff","#00d4aa","#ffa657"]

            sens    = {}
            effects = {}
            for feat, vals in ranges.items():
                preds = []
                for v in vals:
                    cur = baseline.copy()
                    cur[feat] = float(v)
                    if feat == "p1":
                        lo, hi = admissible_p0_bounds(cur["p1"], ys_h)
                        if lo is None: preds.append(np.nan); continue
                        cur["p0"] = np.clip(cur["p0"], lo, hi)
                    preds.append(predict_nn(cur["p0"], cur["p1"],
                                            cur["T_out_C"], cur["T_in_C"],
                                            cur["V_inf"],  cur["q0"]))
                preds = np.array(preds, dtype=float)
                sens[feat] = (vals, preds)
                valid = preds[~np.isnan(preds)]
                effects[feat] = float(np.max(valid) - np.min(valid)) if len(valid) else 0.0

        # Сохраняем результат
        st.session_state.sens_result = {
            "sens": sens, "effects": effects,
            "labels": labels, "colors": colors,
            "baseline": baseline
        }

    # Всегда показываем последний результат sensitivity
    if st.session_state.sens_result is not None:
        s        = st.session_state.sens_result
        sens     = s["sens"]
        effects  = s["effects"]
        labels   = s["labels"]
        colors   = s["colors"]
        baseline = s["baseline"]

        # 6 графиков
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.patch.set_facecolor("#0d1117")
        axes = axes.flatten()
        for idx, (feat, (vals, preds)) in enumerate(sens.items()):
            ax = axes[idx]
            ax.set_facecolor("#161b22")
            ax.plot(vals, preds, color=colors[idx], linewidth=2.5)
            ax.axvline(baseline[feat], color="white", linewidth=1,
                       linestyle="--", alpha=0.5, label="Current")
            ax.set_title(labels[feat], color="#c9d1d9", fontsize=10, pad=6)
            ax.set_ylabel("Q_in (W/m)", color="#8b949e", fontsize=8)
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.grid(True, alpha=0.15, color="#30363d")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        plt.suptitle("Sensitivity Analysis — Effect of Each Parameter on Q_in",
                     color="#c9d1d9", fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature importance bar chart
        st.markdown("#### 📊 Feature Importance (Effect Range)")
        sorted_e   = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
        feat_names = [labels[f] for f, _ in sorted_e]
        feat_vals  = [abs(v) for _, v in sorted_e]

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor("#0d1117")
        ax2.set_facecolor("#161b22")
        bars = ax2.barh(feat_names, feat_vals,
                        color=colors[:len(feat_names)], alpha=0.85, height=0.6)
        for bar, val in zip(bars, feat_vals):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}", va="center", color="#c9d1d9", fontsize=9)
        ax2.set_xlabel("|Max − Min| of Q_in (W/m)", color="#8b949e")
        ax2.set_title("Feature Importance", color="#c9d1d9", pad=8)
        ax2.tick_params(colors="#8b949e")
        for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

else:
    st.info("Load the trained neural network model to run sensitivity analysis.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="color:#8b949e;font-size:0.8rem;text-align:center;">
    Optimizing Insulation Layer for Energy Efficiency in Building Facades · Neural Networks Project
</p>
""", unsafe_allow_html=True)
