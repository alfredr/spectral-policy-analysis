#!/usr/bin/env python3
"""
Spectral Policy Analysis - Streamlit App
Run with: streamlit run spectral_app.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from jpegify import (
    build_kernel,
    cohesion,
    optimize_svd_params,
    permute_matrix,
    pr_curve_from_scores,
    spherical_kmeans,
    stamp_blocks,
    svd_based_group_decomposition,
    svd_reconstruct,
    svd_vs_eig_alignment,
    top_svd,
)

st.set_page_config(page_title="Spectral Policy Analysis", layout="wide")

st.title("Spectral Policy Analysis")

# Sidebar controls
st.sidebar.header("Parameters")

# Generation params
st.sidebar.subheader("Resource Assignment Generation")
m = st.sidebar.number_input("Users (m)", 20, 400, 120, 10)
n = st.sidebar.number_input("Resources (n)", 20, 400, 120, 10)
num_blocks = st.sidebar.number_input("Num groups", 1, 20, 6, 1)
noise = st.sidebar.slider("Noise rate", 0.0, 0.5, 0.0, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 7, 1)

if st.sidebar.button("Regenerate"):
    st.session_state.clear()

# Analysis params
st.sidebar.subheader("Analysis Controls")

kernel = st.sidebar.selectbox("Kernel", ["degree", "cosine", "raw"], index=0)

# Initialize session state for params
if 'j_value' not in st.session_state:
    st.session_state.j_value = 6
if 'tau_value' not in st.session_state:
    st.session_state.tau_value = 0.5

j = st.sidebar.slider("Rank j (SVD components)", 1, 40, st.session_state.j_value, 1, key="j_slider")
tau = st.sidebar.slider("Threshold τ (group membership & reconstruction)", 0.0, 1.0, st.session_state.tau_value, 0.01, key="tau_slider")

# Update session state when sliders change
st.session_state.j_value = j
st.session_state.tau_value = tau

if st.sidebar.button("Find Optimal Params", use_container_width=True):
    st.session_state.optimize_now = True

st.sidebar.subheader("Visualization")
permute_viz = st.sidebar.checkbox(
    "Random permutation",
    value=True,
    help="Randomly permute users/resources to emphasize permutation-invariance"
)

st.sidebar.subheader("SVD Method")
st.sidebar.info(f"Using **randomized SVD** with seed={seed}")


# Cache generation + per-kernel spectral factors
@st.cache_data
def generate_P(m, n, num_blocks, noise, seed, permute):
    P, planted, noise_count = stamp_blocks(
        m,
        n,
        num_blocks=num_blocks,
        min_hw=(max(4, m // 15), max(4, n // 15)),
        max_hw=(max(6, m // 6), max(6, n // 6)),
        density=1.0,
        noise=noise,
        seed=seed,
    )
    if permute:
        P, planted, _, _ = permute_matrix(P, planted, seed=seed)
    return P, planted, noise_count


@st.cache_data
def compute_svd(P, measure, seed):
    M = build_kernel(P, measure=measure)
    U, s, Vt = top_svd(M, k=60, method="randomized", seed=seed)  # keep a decent tail for spectra viz
    return U, s, Vt


@st.cache_data
def compute_alignment(P, measure, k, seed):
    return svd_vs_eig_alignment(P, measure, k, seed=seed)


# Generate data
P, planted, noise_count = generate_P(m, n, num_blocks, noise, seed, permute_viz)

# Compute SVD & alignment
U, s, Vt = compute_svd(P, kernel, seed)
align = compute_alignment(P, kernel, k=min(j, 20), seed=seed)

# Run optimization if requested
if 'optimize_now' in st.session_state and st.session_state.optimize_now:
    with st.spinner("Optimizing parameters..."):
        opt_result = optimize_svd_params(P, U, s, Vt, j_range=(2, 20))
        # Set the optimal values in session state
        st.session_state.j_value = opt_result['best_j']
        st.session_state.tau_value = opt_result['best_tau']
        st.session_state.optimize_now = False
        st.success(f"Optimal parameters set: j={opt_result['best_j']}, τ={opt_result['best_tau']:.2f}, complexity={opt_result['min_complexity']}")
        st.rerun()

# Reconstruction & thresholding
P_svd = svd_reconstruct(P, U, s, Vt, j, measure=kernel)

# PR/threshold metrics
pr = pr_curve_from_scores(P, P_svd)

# Cohesion (cluster users and compute cohesion)
num_clusters = min(num_blocks, j)
user_labels = spherical_kmeans(U[:, :j], k=num_clusters, seed=seed)
clusters = [user_labels == c for c in range(num_clusters)]
cohesion_scores = cohesion(P, clusters, measure="cosine")
avg_cohesion = float(np.mean(cohesion_scores)) if cohesion_scores else 0.0

# Helper for binary stats
def bin_stats(P_pred):
    tp = int(((P_pred == 1) & (P == 1)).sum())
    fp = int(((P_pred == 1) & (P == 0)).sum())
    fn = int(((P_pred == 0) & (P == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return tp, fp, fn, prec, rec

P_recon = (P_svd >= tau).astype(np.uint8)

# Compute binary stats
tp, fp, fn, prec, rec = bin_stats(P_recon)

# SVD-based group decomposition (uses same threshold as SVD reconstruction)
decomp = svd_based_group_decomposition(P, U, s, Vt, j, threshold=tau)

symdiff = (P ^ P_recon).astype(np.uint8)

# ====== Row 1: Original, Reconstruction & Symmetric Difference ======
def viz_diff(P_pred):
    # 0 = correct 0 (white), 1 = correct 1 (black), 2 = diff (red)
    viz = np.zeros_like(P_pred, dtype=float)
    viz[P == 1] = 1.0  # Ground truth 1s -> black
    viz[P_pred != P] = 2.0  # Differences -> red
    return viz

fig1 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        f"Original (assignments={int(P.sum())})",
        f"{kernel} kernel | Rank-{j}",
        f"Symmetric difference (τ={tau:.2f})",
    ),
    horizontal_spacing=0.08,
)

# Original
fig1.add_trace(
    go.Heatmap(
        z=P,
        colorscale="Greys",
        showscale=False,
        hovertemplate="user=%{y}<br>res=%{x}<br>val=%{z}<extra></extra>",
    ),
    row=1,
    col=1,
)
for b in planted:
    fig1.add_shape(
        type="rect",
        x0=b["c0"] - 0.5,
        y0=b["r0"] - 0.5,
        x1=b["c0"] + b["w"] - 0.5,
        y1=b["r0"] + b["h"] - 0.5,
        line=dict(color="red", width=2),
        row=1,
        col=1,
    )

# Reconstruction
fig1.add_trace(
    go.Heatmap(
        z=P_svd,
        colorscale="RdBu_r",
        showscale=True,
        colorbar=dict(title="Value", x=1.02, len=0.5),
        hovertemplate="user=%{y}<br>res=%{x}<br>val=%{z:.3f}<extra></extra>",
    ),
    row=1,
    col=2,
)

# Symmetric difference
fig1.add_trace(
    go.Heatmap(
        z=viz_diff(P_recon),
        colorscale=[[0, "white"], [0.499, "white"], [0.5, "black"], [0.999, "black"], [1, "red"]],
        showscale=False,
        customdata=np.stack([P, P_recon], axis=-1),
        hovertemplate="user=%{y}<br>res=%{x}<br>original=%{customdata[0]}<br>kernel=%{customdata[1]}<extra></extra>",
    ),
    row=1,
    col=3,
)

for c in [1, 2, 3]:
    fig1.update_xaxes(scaleanchor=f"y{c}", scaleratio=1, title_text="resources", row=1, col=c)
    fig1.update_yaxes(title_text="users", row=1, col=c)

fig1.update_layout(height=520, showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# ====== Policy Complexity ======
st.markdown("### Policy Complexity")
base_complexity = n + m + noise_count
fit_complexity = decomp['complexity']
residual_complexity = fit_complexity - base_complexity

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Base Complexity", base_complexity)
with col2:
    st.metric("Base Groups", num_blocks)
with col3:
    st.metric("Base Exceptions", noise_count)
with col4:
    st.metric("Residual Complexity", residual_complexity)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Fit Complexity", fit_complexity)
with col2:
    st.metric("Fit Groups", decomp['num_groups'])
with col3:
    st.metric("Fit Exceptions", decomp['num_exceptions'])

with st.expander("View policy tuples"):
    st.markdown("**User → Group tuples**")
    user_group_text = ", ".join([f"u{u}→g{g}" for u, g in decomp['user_to_group'][:30]])
    if len(decomp['user_to_group']) > 30:
        user_group_text += f" ... ({len(decomp['user_to_group'])} total)"
    st.text(user_group_text)

    st.markdown("**Resource → Group tuples**")
    res_group_text = ", ".join([f"r{r}→g{g}" for r, g in decomp['resource_to_group'][:30]])
    if len(decomp['resource_to_group']) > 30:
        res_group_text += f" ... ({len(decomp['resource_to_group'])} total)"
    st.text(res_group_text)

    st.markdown("**Exceptions**")
    exc_text = ", ".join([f"{op}(u{u}, r{r})" for op, u, r in decomp['exceptions'][:30]])
    if len(decomp['exceptions']) > 30:
        exc_text += f" ... ({len(decomp['exceptions'])} total)"
    st.text(exc_text if exc_text else "(none)")

# ====== Spectral Analysis ======
st.markdown("### Spectral Analysis")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("AUPRC", f"{pr['auprc']:.3f}")
with c2:
    st.metric("Best F1", f"{pr['best_f1']:.3f}")
with c3:
    st.metric("Avg Cohesion", f"{avg_cohesion:.3f}")
with c4:
    st.metric("False Positives", fp)
with c5:
    st.metric("False Negatives", fn)

# ====== Spectra comparison ======
st.markdown("### Spectra: s² vs eigenvalues")
s2 = align["s"] ** 2
wU = align["eig_user_vals"]
wR = align["eig_res_vals"]
kshow = min(len(s2), len(wU), len(wR))
x = np.arange(1, kshow + 1)

fig_spectra = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("s² vs eig(K_u)", "s² vs eig(K_r)"),
    horizontal_spacing=0.12,
)

fig_spectra.add_trace(
    go.Scatter(x=x, y=s2[:kshow], mode="lines+markers", name="s²"), row=1, col=1
)
fig_spectra.add_trace(
    go.Scatter(x=x, y=wU[:kshow], mode="lines+markers", name="eig Ku"), row=1, col=1
)

fig_spectra.add_trace(
    go.Scatter(x=x, y=s2[:kshow], mode="lines+markers", name="s²"), row=1, col=2
)
fig_spectra.add_trace(
    go.Scatter(x=x, y=wR[:kshow], mode="lines+markers", name="eig Kr"), row=1, col=2
)

fig_spectra.update_layout(height=360, showlegend=True)
st.plotly_chart(fig_spectra, use_container_width=True)

# Diagnostics: check for users/resources without group membership
users_with_groups = set(u for u, g in decomp['user_to_group'])
resources_with_groups = set(r for r, g in decomp['resource_to_group'])
orphan_users = m - len(users_with_groups)
orphan_resources = n - len(resources_with_groups)

if orphan_users > 0 or orphan_resources > 0:
    st.warning(f"⚠️ {orphan_users} users and {orphan_resources} resources have no group membership (all below threshold τ={tau:.2f})")

# Show comparison with FP+FN
tp, fp, fn, prec, rec = bin_stats(P_recon)
st.caption(f"(SVD reconstruction has {fp + fn} errors = FP+FN)")
