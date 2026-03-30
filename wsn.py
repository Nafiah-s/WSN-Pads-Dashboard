"""
=============================================================================
WSN PADS Simulation — Streamlit Dashboard
=============================================================================
Run command:
    streamlit run wsn_streamlit_app.py

Install requirements:
    pip install streamlit numpy pandas matplotlib
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WSN PADS Simulation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #1565C0;
        text-align: center; padding: 10px 0 4px 0;
    }
    .sub-title {
        font-size: 1rem; color: #666;
        text-align: center; margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1A237E;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 5px; margin: 18px 0 12px 0;
    }
    .card-blue   { background: linear-gradient(135deg,#E3F2FD,#BBDEFB);
                   border-left: 5px solid #1565C0; border-radius:10px; padding:14px 16px; }
    .card-orange { background: linear-gradient(135deg,#FFF3E0,#FFE0B2);
                   border-left: 5px solid #E65100; border-radius:10px; padding:14px 16px; }
    .card-green  { background: linear-gradient(135deg,#E8F5E9,#C8E6C9);
                   border-left: 5px solid #2E7D32; border-radius:10px; padding:14px 16px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📡 WSN PADS Simulation Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Priority-Aware Dynamic Scheduling vs Random vs Static '
    '— Final Year Engineering Project</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")

    st.markdown("### 🌐 Environment")
    num_nodes    = st.slider("Number of Nodes",        10, 150, 50, 5)
    num_rounds   = st.slider("Simulation Rounds",      20, 300, 100, 10)
    active_ratio = st.slider("Active Ratio (%)",       10, 80,  50, 5) / 100
    seed         = st.number_input("Random Seed",      value=42, step=1)

    st.markdown("### ⚡ Energy")
    initial_energy = st.slider("Initial Energy (J)",      0.5, 5.0,  2.0, 0.5)
    energy_active  = st.slider("Active Drain (J/round)",  0.01, 0.15, 0.05, 0.01)
    energy_sleep   = st.slider("Sleep Drain (J/round)",   0.001, 0.02, 0.005, 0.001)

    st.markdown("### 📡 Sensing")
    sensing_radius = st.slider("Sensing Radius (m)",       5, 30, 15, 1)
    grid_res       = st.selectbox("Grid Resolution (m)",   [1, 2, 5], index=1)

    st.markdown("### 🧠 PADS Weights")
    w_energy  = st.slider("Energy Weight",   0.1, 0.9, 0.6, 0.05)
    w_history = st.slider("History Weight",  0.1, 0.9, 0.4, 0.05)

    st.markdown("### 🎯 Priority Zones")
    zone_high = st.slider("High Zone Radius (m)",    5, 30, 20, 1)
    zone_med  = st.slider("Medium Zone Radius (m)",  25, 60, 45, 1)

    st.markdown("---")
    run_btn = st.button("🚀 Run Simulation", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD CONFIG DICT
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "area_width":       100,
    "area_height":      100,
    "num_nodes":        num_nodes,
    "num_rounds":       num_rounds,
    "grid_resolution":  grid_res,
    "seed":             int(seed),
    "initial_energy":   initial_energy,
    "energy_active":    energy_active,
    "energy_sleep":     energy_sleep,
    "sensing_radius":   sensing_radius,
    "active_ratio":     active_ratio,
    "weight_energy":    w_energy,
    "weight_history":   w_history,
    "zone_high_radius": zone_high,
    "zone_med_radius":  zone_med,
    "priority_high":    1.5,
    "priority_med":     1.0,
    "priority_low":     0.7,
}

COLORS = {"pads": "#1565C0", "random": "#E65100", "static": "#2E7D32"}
LABELS = {
    "pads":   "PADS (Proposed)",
    "random": "Random (Baseline)",
    "static": "Static (Baseline)",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────────────────────────────────────

def create_nodes(cfg):
    rng = np.random.default_rng(cfg["seed"])
    return [{
        "id": i,
        "x":  rng.uniform(0, cfg["area_width"]),
        "y":  rng.uniform(0, cfg["area_height"]),
        "initial_energy":   cfg["initial_energy"],
        "energy":           cfg["initial_energy"],
        "status":           "Sleep",
        "activation_count": 0,
    } for i in range(cfg["num_nodes"])]


def build_grid(cfg):
    r  = cfg["grid_resolution"]
    xs = np.arange(0, cfg["area_width"]  + r, r)
    ys = np.arange(0, cfg["area_height"] + r, r)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def get_priority_weight(node, cfg):
    cx   = cfg["area_width"]  / 2
    cy   = cfg["area_height"] / 2
    dist = np.hypot(node["x"] - cx, node["y"] - cy)
    if dist <= cfg["zone_high_radius"]: return cfg["priority_high"]
    if dist <= cfg["zone_med_radius"]:  return cfg["priority_med"]
    return cfg["priority_low"]


def get_zone_label(node, cfg):
    cx   = cfg["area_width"]  / 2
    cy   = cfg["area_height"] / 2
    dist = np.hypot(node["x"] - cx, node["y"] - cy)
    if dist <= cfg["zone_high_radius"]: return "High"
    if dist <= cfg["zone_med_radius"]:  return "Medium"
    return "Low"


def compute_coverage(nodes, grid_points, cfg):
    active = np.array([[n["x"], n["y"]] for n in nodes if n["status"] == "Active"])
    if len(active) == 0:
        return 0.0
    covered = np.zeros(len(grid_points), dtype=bool)
    for pos in active:
        covered |= (np.linalg.norm(grid_points - pos, axis=1) <= cfg["sensing_radius"])
    return 100.0 * covered.sum() / len(grid_points)


def schedule_pads(nodes, cfg):
    alive   = [n for n in nodes if n["status"] != "Dead"]
    n_act   = max(1, int(len(alive) * cfg["active_ratio"]))
    max_act = max((n["activation_count"] for n in alive), default=1) or 1
    for n in alive:
        n["_score"] = (
            cfg["weight_energy"]  * (n["energy"] / n["initial_energy"])
          - cfg["weight_history"] * (n["activation_count"] / max_act)
          + 0.2 * get_priority_weight(n, cfg)
        )
    top = sorted(alive, key=lambda x: x["_score"], reverse=True)[:n_act]
    active_ids = {n["id"] for n in top}
    for n in nodes:
        if n["status"] == "Dead": continue
        if n["id"] in active_ids:
            n["status"] = "Active"
            n["activation_count"] += 1
        else:
            n["status"] = "Sleep"
    return nodes


def schedule_random(nodes, cfg):
    alive = [n for n in nodes if n["status"] != "Dead"]
    n_act = max(1, int(len(alive) * cfg["active_ratio"]))
    active_ids = {n["id"] for n in random.sample(alive, min(n_act, len(alive)))}
    for n in nodes:
        if n["status"] == "Dead": continue
        n["status"] = "Active" if n["id"] in active_ids else "Sleep"
    return nodes


def schedule_static(nodes, static_ids):
    for n in nodes:
        if n["status"] == "Dead": continue
        n["status"] = "Active" if n["id"] in static_ids else "Sleep"
    return nodes


def update_energies(nodes, cfg):
    for n in nodes:
        if n["status"] == "Dead": continue
        n["energy"] -= cfg["energy_active"] if n["status"] == "Active" else cfg["energy_sleep"]
        if n["energy"] <= 0:
            n["energy"] = 0.0
            n["status"] = "Dead"
    return nodes


def run_simulation(strategy, cfg):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    nodes       = create_nodes(cfg)
    grid_points = build_grid(cfg)
    static_ids  = set()

    if strategy == "static":
        alive      = list(nodes)
        n_act      = max(1, int(len(alive) * cfg["active_ratio"]))
        static_ids = {n["id"] for n in random.sample(alive, n_act)}

    results = []
    for rnd in range(1, cfg["num_rounds"] + 1):
        if strategy == "pads":
            nodes = schedule_pads(nodes, cfg)
        elif strategy == "random":
            nodes = schedule_random(nodes, cfg)
        else:
            nodes = schedule_static(nodes, static_ids)

        nodes        = update_energies(nodes, cfg)
        alive_nodes  = [n for n in nodes if n["status"] != "Dead"]
        coverage_pct = compute_coverage(nodes, grid_points, cfg)
        avg_energy   = float(np.mean([n["energy"] for n in alive_nodes])) if alive_nodes else 0.0

        results.append({
            "round":        rnd,
            "coverage_pct": round(coverage_pct, 4),
            "alive_nodes":  len(alive_nodes),
            "avg_energy":   round(avg_energy, 6),
        })
        if not alive_nodes:
            break

    return pd.DataFrame(results), nodes


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_line(results_dict, metric, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s, df in results_dict.items():
        ax.plot(df["round"], df[metric], color=COLORS[s],
                label=LABELS[s], linewidth=2.2, alpha=0.9)
    ax.set_xlabel("Simulation Round", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, CFG["num_rounds"])
    fig.tight_layout()
    return fig


def plot_combined(results_dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("WSN Scheduling Strategy Comparison", fontsize=14, fontweight="bold")
    panels = [
        ("coverage_pct", "Coverage (%)",   "Coverage vs Rounds"),
        ("alive_nodes",  "Alive Nodes",    "Alive Nodes vs Rounds"),
        ("avg_energy",   "Avg Energy (J)", "Avg Energy vs Rounds"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for s, df in results_dict.items():
            ax.plot(df["round"], df[metric], color=COLORS[s],
                    label=LABELS[s], linewidth=2.0, alpha=0.9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(1, CFG["num_rounds"])
    handles = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_deployment(cfg):
    nodes = create_nodes(cfg)
    cx, cy = cfg["area_width"] / 2, cfg["area_height"] / 2
    zone_colors = {"High": "#E53935", "Medium": "#FB8C00", "Low": "#43A047"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for n in nodes:
        lbl = get_zone_label(n, cfg)
        c   = zone_colors[lbl]
        ax.scatter(n["x"], n["y"], color=c, s=60, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.add_patch(plt.Circle((n["x"], n["y"]), cfg["sensing_radius"],
                                color=c, fill=False, alpha=0.12, linewidth=0.5))
    for r, ls in [(cfg["zone_high_radius"], "--"), (cfg["zone_med_radius"], ":")]:
        ax.add_patch(plt.Circle((cx, cy), r, color="gray",
                                fill=False, linestyle=ls, linewidth=1.5))
    handles = [mpatches.Patch(color=zone_colors[z], label=f"{z} Priority")
               for z in ["High", "Medium", "Low"]]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_xlim(0, cfg["area_width"])
    ax.set_ylim(0, cfg["area_height"])
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_title("Node Deployment + Priority Zones", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bar(results_dict):
    strategies = list(results_dict.keys())
    means = [results_dict[s]["coverage_pct"].mean() for s in strategies]
    maxes = [results_dict[s]["coverage_pct"].max()  for s in strategies]
    x   = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x - 0.2, means, 0.35, label="Mean Coverage",
                  color=[COLORS[s] for s in strategies], alpha=0.85)
    ax.bar(x + 0.2, maxes, 0.35, label="Max Coverage",
           color=[COLORS[s] for s in strategies], alpha=0.4,
           edgecolor=[COLORS[s] for s in strategies], linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in strategies], fontsize=9)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Mean vs Max Coverage Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_energy_map(final_nodes_dict, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (strategy, nodes) in zip(axes, final_nodes_dict.items()):
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]
        es = [n["energy"] for n in nodes]
        sc = ax.scatter(xs, ys, c=es, cmap="RdYlGn",
                        vmin=0, vmax=cfg["initial_energy"],
                        s=80, edgecolors="gray", linewidths=0.4)
        plt.colorbar(sc, ax=ax, label="Energy (J)")
        ax.set_title(f"{LABELS[strategy]}\nFinal Energy State",
                     fontsize=10, fontweight="bold")
        ax.set_xlim(0, cfg["area_width"])
        ax.set_ylim(0, cfg["area_height"])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE (before simulation runs)
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "### 🧠 PADS\n"
            "Scores every node on **energy + history + zone priority** "
            "→ activates top-scoring 50% each round."
        )
    with c2:
        st.warning(
            "### 🎲 Random\n"
            "Randomly activates 50% of alive nodes — "
            "no energy awareness, no zone logic."
        )
    with c3:
        st.success(
            "### 🔒 Static\n"
            "Same fixed nodes active all rounds — "
            "traditional naive approach."
        )

    st.markdown("---")
    st.markdown('<div class="section-header">📌 Node Deployment Preview</div>',
                unsafe_allow_html=True)
    fig = plot_deployment(CFG)
    st.pyplot(fig, use_container_width=False)
    plt.close()
    st.caption("👈 Adjust parameters in the sidebar → Click **🚀 Run Simulation**")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS (after simulation runs)
# ─────────────────────────────────────────────────────────────────────────────
else:
    # Run all 3 strategies
    progress    = st.progress(0, text="Initialising...")
    results     = {}
    final_nodes = {}

    for idx, strategy in enumerate(["pads", "random", "static"]):
        progress.progress(idx * 33, text=f"Running {LABELS[strategy]}...")
        df, nodes_end         = run_simulation(strategy, CFG)
        results[strategy]     = df
        final_nodes[strategy] = nodes_end

    progress.progress(100, text="✅ Simulation complete!")

    # Winner banner
    best = max(results, key=lambda s: results[s]["coverage_pct"].mean())
    st.success(
        f"🏆 **Best Strategy: {LABELS[best]}** — "
        f"Mean Coverage **{results[best]['coverage_pct'].mean():.2f}%**"
    )

    # Summary cards
    st.markdown('<div class="section-header">📊 Performance Summary</div>',
                unsafe_allow_html=True)
    card_css = ["card-blue", "card-orange", "card-green"]
    cols     = st.columns(3)
    for col, strategy, css in zip(cols, ["pads", "random", "static"], card_css):
        df = results[strategy]
        with col:
            st.markdown(f"""
            <div class="{css}">
                <b style="font-size:1rem">{LABELS[strategy]}</b>
                <hr style="margin:6px 0">
                🎯 Mean Coverage : <b>{df['coverage_pct'].mean():.2f}%</b><br>
                🔝 Max Coverage  : <b>{df['coverage_pct'].max():.2f}%</b><br>
                💀 Final Alive   : <b>{int(df['alive_nodes'].iloc[-1])} nodes</b><br>
                🔁 Rounds Run    : <b>{len(df)}</b><br>
                ⚡ Final Energy  : <b>{df['avg_energy'].iloc[-1]:.4f} J</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Coverage",
        "💀 Alive Nodes",
        "⚡ Avg Energy",
        "📊 Combined",
        "📊 Bar Chart",
        "🗺️ Node Maps",
        "📋 Raw Data",
    ])

    # ── Tab 1: Coverage ───────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Coverage % vs Simulation Rounds</div>',
                    unsafe_allow_html=True)
        fig = plot_line(results, "coverage_pct", "Coverage (%)",
                        "Coverage vs Simulation Rounds")
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Chart", fig_to_bytes(fig),
                           "coverage_vs_rounds.png", "image/png")
        plt.close()
        pads_m   = results["pads"]["coverage_pct"].mean()
        rand_m   = results["random"]["coverage_pct"].mean()
        stat_m   = results["static"]["coverage_pct"].mean()
        st.info(
            f"**PADS** achieves **{pads_m:.2f}%** mean coverage vs "
            f"Random **{rand_m:.2f}%** and Static **{stat_m:.2f}%**. "
            f"PADS beats Random by **{pads_m - rand_m:.2f}%** "
            f"and Static by **{pads_m - stat_m:.2f}%**."
        )

    # ── Tab 2: Alive Nodes ────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Alive Nodes vs Simulation Rounds</div>',
                    unsafe_allow_html=True)
        fig = plot_line(results, "alive_nodes", "Number of Alive Nodes",
                        "Alive Nodes vs Simulation Rounds")
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Chart", fig_to_bytes(fig),
                           "alive_nodes_vs_rounds.png", "image/png")
        plt.close()
        st.info(
            "Static may show more alive nodes at the end because its unactivated "
            "nodes never spend energy — but they also contribute nothing to coverage. "
            "PADS distributes load fairly across all nodes."
        )

    # ── Tab 3: Avg Energy ─────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Average Node Energy vs Simulation Rounds</div>',
                    unsafe_allow_html=True)
        fig = plot_line(results, "avg_energy", "Avg Energy (J)",
                        "Average Node Energy vs Simulation Rounds")
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Chart", fig_to_bytes(fig),
                           "avg_energy_vs_rounds.png", "image/png")
        plt.close()

        st.markdown("#### 🔋 Energy Drain Summary")
        e_cols = st.columns(3)
        for col, s in zip(e_cols, ["pads", "random", "static"]):
            final_e = results[s]["avg_energy"].iloc[-1]
            drained = CFG["initial_energy"] - final_e
            col.metric(
                label=LABELS[s],
                value=f"{final_e:.4f} J remaining",
                delta=f"-{drained:.4f} J drained",
                delta_color="inverse",
            )

    # ── Tab 4: Combined Dashboard ─────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Combined 3-Panel Dashboard</div>',
                    unsafe_allow_html=True)
        fig = plot_combined(results)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Combined Chart", fig_to_bytes(fig),
                           "combined_comparison.png", "image/png")
        plt.close()

    # ── Tab 5: Bar Chart ──────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Mean vs Max Coverage Bar Chart</div>',
                    unsafe_allow_html=True)
        fig = plot_bar(results)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Bar Chart", fig_to_bytes(fig),
                           "bar_comparison.png", "image/png")
        plt.close()

    # ── Tab 6: Node Maps ──────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">Node Deployment Map</div>',
                    unsafe_allow_html=True)
        fig = plot_deployment(CFG)
        st.pyplot(fig, use_container_width=False)
        st.download_button("⬇️ Download Deployment Map", fig_to_bytes(fig),
                           "node_deployment.png", "image/png")
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-header">Final Energy State (end of simulation)</div>',
                    unsafe_allow_html=True)
        st.caption("🟢 Green = high energy remaining  |  🔴 Red = depleted")
        fig = plot_energy_map(final_nodes, CFG)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Energy Map", fig_to_bytes(fig),
                           "energy_map.png", "image/png")
        plt.close()

    # ── Tab 7: Raw Data ───────────────────────────────────────────────────────
    with tab7:
        st.markdown('<div class="section-header">Raw Simulation Data</div>',
                    unsafe_allow_html=True)

        choice = st.selectbox(
            "Select Strategy",
            options=["pads", "random", "static"],
            format_func=lambda s: LABELS[s],
        )
        df_show = results[choice]
        st.dataframe(df_show, use_container_width=True, height=380)
        st.download_button(
            f"⬇️ Download {LABELS[choice]} CSV",
            df_show.to_csv(index=False).encode("utf-8"),
            f"{choice}_results.csv",
            "text/csv",
        )

        st.markdown("---")
        st.markdown("#### 📊 Descriptive Statistics")
        st.dataframe(df_show.describe().round(4), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📥 Download All 3 CSV Files")
        dl_cols = st.columns(3)
        for col, s in zip(dl_cols, ["pads", "random", "static"]):
            col.download_button(
                f"⬇️ {LABELS[s]}",
                results[s].to_csv(index=False).encode("utf-8"),
                f"{s}_results.csv",
                "text/csv",
            )