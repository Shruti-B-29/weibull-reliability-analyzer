import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import weibull_min
from scipy.special import gamma

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Weibull Reliability Analyzer",
    page_icon="⚙️",
    layout="wide"
)

# ── Helper functions ──────────────────────────────────────────
def weibull_pdf(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1) * np.exp(-(t / eta) ** beta)

def weibull_cdf(t, beta, eta):
    return 1 - np.exp(-(t / eta) ** beta)

def weibull_reliability(t, beta, eta):
    return np.exp(-(t / eta) ** beta)

def weibull_hazard(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1)

def b_life(x_percent, beta, eta):
    return eta * (-np.log(1 - x_percent / 100)) ** (1 / beta)

def mttf(beta, eta):
    return eta * gamma(1 + 1 / beta)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title(" Weibull Parameters")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    [" Explore Parameters", " Upload Your Data"],
    help="Manually explore Weibull behaviour or fit to real failure data"
)

if mode == "Explore Parameters":
    beta = st.sidebar.slider("Shape parameter β", 0.5, 8.0, 4.41, 0.01,
                              help="β < 1: early failures | β = 1: random | β > 1: wear-out")
    eta  = st.sidebar.slider("Scale parameter η (cycles)", 50.0, 500.0, 225.0, 1.0,
                              help="Cycle at which 63.2% of units have failed")
    t_max = st.sidebar.slider("Time axis max (cycles)", 100, 800, 400, 10)
    uploaded = None

else:
    uploaded = st.sidebar.file_uploader(
        "Upload CMAPSS train_FD001.txt",
        type=["txt", "csv"],
        help="NASA CMAPSS format: space-separated, columns = engine_id, cycle, op1..3, s1..21"
    )
    t_max = st.sidebar.slider("Time axis max (cycles)", 100, 800, 400, 10)
    beta, eta = 4.41, 225.0  # defaults until data loads

# ── Main title ────────────────────────────────────────────────
st.title(" Weibull Reliability Analyzer")
st.markdown("**NASA CMAPSS Turbofan Engine Dataset | Failure Analysis & Reliability Curves**")
st.markdown("---")

# ── Data upload flow ──────────────────────────────────────────
failure_times = None
r2_2param = None
r2_3param = None
gamma_val  = None

if mode == " Upload Your Data" and uploaded:
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(uploaded, sep=r'\s+', header=None, names=cols)
    failure_times = df.groupby('engine_id')['cycle'].max().values

    # Fit 2-param
    shape2, loc2, scale2 = weibull_min.fit(failure_times, floc=0)
    beta, eta = shape2, scale2

    # Fit 3-param
    shape3, loc3, scale3 = weibull_min.fit(failure_times)
    gamma_val = loc3

    # R² for both
    n = len(failure_times)
    t_s = np.sort(failure_times)
    mr = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

    x2 = np.log(t_s);           y2 = np.log(-np.log(1 - mr))
    c2 = np.polyfit(x2, y2, 1); r2_2param = 1 - np.sum((y2 - np.polyval(c2, x2))**2) / np.sum((y2 - y2.mean())**2)

    x3 = np.log(t_s - loc3);    y3 = np.log(-np.log(1 - mr))
    c3 = np.polyfit(x3, y3, 1); r2_3param = 1 - np.sum((y3 - np.polyval(c3, x3))**2) / np.sum((y3 - y3.mean())**2)

    st.success(f" Loaded {len(failure_times)} engines · Failure range: {failure_times.min()}–{failure_times.max()} cycles")

# ── Metrics row ───────────────────────────────────────────────
MTTF_val = mttf(beta, eta)
B10 = b_life(10, beta, eta)
B50 = b_life(50, beta, eta)
B90 = b_life(90, beta, eta)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Shape β",  f"{beta:.4f}")
col2.metric("Scale η",  f"{eta:.1f}")
col3.metric("MTTF",     f"{MTTF_val:.1f} cycles")
col4.metric("B10 Life", f"{B10:.1f} cycles")
col5.metric("B50 Life", f"{B50:.1f} cycles")

st.markdown("---")

# ── Four reliability curves ───────────────────────────────────
t = np.linspace(0.1, t_max, 1000)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("PDF — Failure Density",
                    "CDF — Cumulative Failure Probability",
                    "R(t) — Reliability Function",
                    "h(t) — Hazard Rate"),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

fig.add_trace(go.Scatter(x=t, y=weibull_pdf(t, beta, eta),
    fill='tozeroy', line=dict(color='steelblue', width=2), name='PDF'), row=1, col=1)

fig.add_trace(go.Scatter(x=t, y=weibull_cdf(t, beta, eta),
    line=dict(color='tomato', width=2), name='CDF'), row=1, col=2)
fig.add_hline(y=0.632, line_dash="dash", line_color="gray",
    annotation_text=f"63.2% at η={eta:.0f}", row=1, col=2)

fig.add_trace(go.Scatter(x=t, y=weibull_reliability(t, beta, eta),
    fill='tozeroy', line=dict(color='seagreen', width=2), name='R(t)'), row=2, col=1)
for bx, col, name in [(B10,'tomato','B10'), (B50,'orange','B50'), (B90,'steelblue','B90')]:
    fig.add_vline(x=bx, line_dash="dot", line_color=col,
        annotation_text=name, row=2, col=1)

fig.add_trace(go.Scatter(x=t, y=weibull_hazard(t, beta, eta),
    line=dict(color='darkorange', width=2), name='h(t)'), row=2, col=2)

fig.update_layout(height=580, showlegend=False,
    title_text=f"Weibull Reliability Curves  |  β={beta:.4f}, η={eta:.4f}",
    title_font_size=14)
st.plotly_chart(fig, use_container_width=True)

# ── Goodness of fit (only when data uploaded) ─────────────────
if failure_times is not None:
    st.markdown("---")
    st.subheader(" Goodness of Fit — Weibull Probability Plot")

    n  = len(failure_times)
    ts = np.sort(failure_times)
    mr = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    x2 = np.log(ts);           y_pts = np.log(-np.log(1 - mr))
    x3 = np.log(ts - gamma_val)

    c2    = np.polyfit(x2, y_pts, 1)
    c3    = np.polyfit(x3, y_pts, 1)
    xl2   = np.linspace(x2.min(), x2.max(), 200)
    xl3   = np.linspace(x3.min(), x3.max(), 200)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x2, y=y_pts, mode='markers',
        marker=dict(color='steelblue', size=7, opacity=0.7), name='Data'))
    fig2.add_trace(go.Scatter(x=xl2, y=np.polyval(c2, xl2),
        line=dict(color='tomato', width=2), name=f'2-param fit  R²={r2_2param:.4f}'))
    fig2.add_trace(go.Scatter(x=xl3, y=np.polyval(c3, xl3),
        line=dict(color='seagreen', width=2, dash='dash'),
        name=f'3-param fit  R²={r2_3param:.4f}'))

    fig2.update_layout(
        height=420,
        xaxis_title='ln(t)',
        yaxis_title='ln(−ln(1−F))',
        title=f'Weibull Probability Plot  |  γ = {gamma_val:.2f} cycles',
        legend=dict(x=0.02, y=0.97)
    )
    st.plotly_chart(fig2, use_container_width=True)

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("2-param R²", f"{r2_2param:.4f}")
    gc2.metric("3-param R²", f"{r2_3param:.4f}")
    gc3.metric("Threshold γ", f"{gamma_val:.2f} cycles")

# ── Interpretation box ────────────────────────────────────────
st.markdown("---")
st.subheader("Interpretation")

if beta < 1:
    failure_mode = "**Infant Mortality** — Early failures dominating. Review manufacturing/QC process."
elif beta == 1:
    failure_mode = "**Random Failures** — Constant hazard rate. Failures are memoryless."
elif beta < 4:
    failure_mode = "**Early Wear-out** — Increasing failure rate. Preventive maintenance recommended."
else:
    failure_mode = "**Advanced Wear-out** — Strongly increasing failure rate. Schedule maintenance before B10 life."

st.markdown(failure_mode)
st.markdown(f"""
| Metric | Value | Meaning |
|---|---|---|
| β = {beta:.4f} | Shape | Wear-out failure mode confirmed |
| η = {eta:.1f} | Scale | 63.2% of units fail by this cycle |
| MTTF = {MTTF_val:.1f} | Mean life | Average engine lifetime |
| B10 = {B10:.1f} | Design life | Schedule maintenance before this cycle |
| B90 = {B90:.1f} | Extended life | 10% of engines survive past this point |
""")

st.markdown("---")
st.caption("Built with Python · scipy · Streamlit · Plotly  |  Dataset: NASA CMAPSS FD001")