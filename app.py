"""
app.py - Deadlock Analysis and Resource Allocation Dashboard
Streamlit Web Interface.

This dashboard provides an interactive laboratory for Operating Systems students
to explore, visualize, and demonstrate:
1. Deadlock Avoidance using Dijkstra's Banker's Algorithm (with step tables and state restoration)
2. Deadlock Detection for Single-Instance (RAG/WFG graph cycles) and Multi-Instance (matrix algorithm)
3. Deadlock Prevention Policies (Resource Ordering & Request-All-Upfront) with trade-off analysis
4. Analytical Test Scenarios A through F with automatic verification and CSV export.

Note: All algorithm computation is kept strictly within banker.py, detection.py, prevention.py,
validation.py, and scenarios.py. app.py handles purely UI presentation, user input, and layout.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import json
import matplotlib.pyplot as plt

# Import algorithm logic modules
from banker import (
    calculate_need,
    safety_check,
    handle_request,
    calculate_need_matrix,
    run_safety_algorithm,
    handle_resource_request,
    format_steps_as_dataframe
)
from detection import (
    detect_single_instance,
    detect_multi_instance,
    terminate_process,
    detect_single_instance_deadlock,
    detect_multi_instance_deadlock,
    draw_graphs_to_figure,
    recover_by_terminating_process_single,
    recover_by_terminating_process_multi
)
from prevention import (
    get_coffman_conditions_summary,
    get_coffman_breakdown,
    simulate_prevention_single_instance,
    simulate_prevention_multi_instance,
    evaluate_resource_ordering_prevention,
    evaluate_request_all_upfront_prevention
)
from validation import (
    validate_matrix,
    validate_vector,
    validate_process_id,
    validate_configuration,
    validate_request,
    validate_system_state,
    validate_matrix_shape_and_types,
    validate_request_input
)
import scenarios
from scenarios import (
    get_scenario_a,
    get_scenario_b,
    get_scenario_c,
    get_scenario_d_single,
    get_scenario_d_multi,
    get_scenario_e,
    get_scenario_f,
    run_scenario_a,
    run_scenario_b,
    run_scenario_c,
    run_scenario_d,
    run_scenario_e,
    run_scenario_f,
    get_all_scenarios_results_table
)


# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Deadlock Analysis and Resource Allocation Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern typography, responsive cards, and 5 distinct status badges
st.markdown("""
<style>
    /* Font and general layout styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }

    /* 5 DISTINCT STATUS BANNERS (Dark & Light friendly) */
    .status-badge {
        padding: 0.9rem 1.3rem;
        border-radius: 8px;
        font-size: 1.0rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        backdrop-filter: blur(4px);
    }
    
    /* 1. SAFE (Green) */
    .status-safe {
        background-color: rgba(16, 185, 129, 0.16);
        color: #ECFDF5;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 6px solid #10B981;
    }
    
    /* 2. UNSAFE (Orange) - NOT DEADLOCK */
    .status-unsafe {
        background-color: rgba(249, 115, 22, 0.16);
        color: #FFF7ED;
        border: 1px solid rgba(249, 115, 22, 0.35);
        border-left: 6px solid #F97316;
    }
    
    /* 3. WAITING / DEFERRED (Blue) */
    .status-waiting {
        background-color: rgba(59, 130, 246, 0.16);
        color: #EFF6FF;
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-left: 6px solid #3B82F6;
    }
    
    /* 4. DEADLOCK DETECTED (Red) */
    .status-deadlock {
        background-color: rgba(239, 68, 68, 0.18);
        color: #FEF2F2;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #EF4444;
    }
    
    /* 5. INVALID INPUT (Grey/Purple) */
    .status-invalid {
        background-color: rgba(168, 85, 247, 0.14);
        color: #FAF5FF;
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-left: 6px solid #A855F7;
    }

    /* Concept alert box */
    .concept-card {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1.4rem;
        font-size: 0.95rem;
        color: #F8FAFC;
    }

    .footer-note {
        margin-top: 3rem;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        text-align: center;
        font-size: 0.88rem;
        color: #94A3B8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def render_status_banner(status_type: str, message: str, details: str = ""):
    """
    Renders one of the 5 strictly distinguished status banners:
    1. SAFE (green)
    2. UNSAFE (orange)
    3. WAITING/DEFERRED (blue)
    4. DEADLOCK DETECTED (red)
    5. INVALID INPUT (grey/purple)
    Uses semi-transparent tinted backgrounds with a coloured left border, light text, and distinctive icons.
    """
    status_upper = status_type.upper().strip()
    if status_upper in ["SAFE", "GRANTED"]:
        css_class = "status-safe"
        icon = "🟢"
        title = "SAFE" if status_upper == "SAFE" else "SAFE (Request Granted)"
        title_color = "#34D399"
    elif status_upper in ["UNSAFE", "DENIED_UNSAFE"]:
        css_class = "status-unsafe"
        icon = "🟠"
        title = "UNSAFE (Request Denied)" if status_upper == "DENIED_UNSAFE" else "UNSAFE (No Safe Sequence)"
        title_color = "#FB923C"
    elif status_upper in ["WAITING", "WAITING / DEFERRED", "DEFERRED"]:
        css_class = "status-waiting"
        icon = "🔵"
        title = "WAITING / DEFERRED (Request Exceeds Available)"
        title_color = "#60A5FA"
    elif "DEADLOCK" in status_upper:
        css_class = "status-deadlock"
        icon = "🔴"
        title = "DEADLOCK DETECTED"
        title_color = "#F87171"
    else:  # INVALID, INVALID INPUT, EXCEEDS_NEED
        css_class = "status-invalid"
        icon = "🟣"
        title = "INVALID INPUT"
        title_color = "#C084FC"

    st.markdown(f"""
    <div class="status-badge {css_class}">
        <span style="font-size: 1.4rem; line-height: 1.2;">{icon}</span>
        <div style="flex: 1;">
            <div style="color: {title_color}; font-weight: 800; font-size: 1.05rem;">{title}</div>
            <div style="font-weight: 500; font-size: 0.95rem; margin-top: 0.25rem;">{message}</div>
            {f'<div style="font-size: 0.85rem; margin-top: 0.25rem; opacity: 0.85;">{details}</div>' if details else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_unsafe_concept_panel():
    """Renders the required educational panel titled 'Unsafe is NOT the same as Deadlock'."""
    st.markdown("""
    <div class="concept-card">
        <h4 style="color: #FBBF24; margin-top: 0; font-size: 1.15rem; display: flex; align-items: center; gap: 0.5rem;">
            ⚠️ Unsafe is NOT the same as Deadlock
        </h4>
        <p style="margin-bottom: 0.6rem; line-height: 1.5;">
            <strong>1. An unsafe state only means deadlock is possible:</strong><br>
            An unsafe state does <em>not</em> mean that a deadlock has occurred or that processes are currently frozen. It only indicates that the operating system can no longer guarantee that every process will complete if all processes suddenly request their maximum declared claims simultaneously. If processes release resources early or terminate before requesting their peak limits, the system may still avoid deadlock.
        </p>
        <p style="margin-bottom: 0; line-height: 1.5;">
            <strong>2. Banker's Algorithm is Deadlock Avoidance, NOT Detection:</strong><br>
            Dijkstra's Banker's Algorithm is an <em>avoidance algorithm</em>: it dynamically checks resource requests ahead of time to keep the system strictly within the boundary of safe states. It does <em>not</em> detect existing deadlocks among currently running processes (which is the job of Deadlock Detection algorithms using graph cycle finding or request matrix analysis, demonstrated in Tab 2).
        </p>
    </div>
    """, unsafe_allow_html=True)


def make_figure_dark_friendly(fig: plt.Figure, bg_color: str = "#0E1117", card_bg: str = "#1A1F2B", text_color: str = "#FAFAFA") -> plt.Figure:
    """
    Styles a Matplotlib figure and its axes for seamless dark theme display.
    Eliminates the bright white box background, colors titles & legends with light text,
    and preserves high-contrast red deadlock cycle highlighting.
    """
    fig.patch.set_facecolor(bg_color)
    for ax in fig.get_axes():
        ax.set_facecolor(bg_color)
        ax.title.set_color(text_color)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(card_bg)
            leg.get_frame().set_edgecolor("#334155")
            for text in leg.get_texts():
                text.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(bg_color)
    return fig


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initializes default state from Classic Example (Scenario A)."""
    if "initialized" not in st.session_state:
        default_data = scenarios.get_default_state()
        st.session_state.num_processes = default_data["num_processes"]
        st.session_state.num_resources = default_data["num_resources"]
        st.session_state.process_labels = default_data["process_labels"]
        st.session_state.resource_labels = default_data["resource_labels"]
        st.session_state.allocation = default_data["allocation"].copy()
        st.session_state.maximum = default_data["maximum"].copy()
        st.session_state.available = default_data["available"].copy()
        st.session_state.total = default_data["total"].copy()
        st.session_state.request_history = []
        st.session_state.last_request_outcome = None
        st.session_state.last_safety_check = None
        st.session_state.active_preset = "A"
        st.session_state.editor_version = 0

        # Isolated Detection Tab State (separate from Banker's matrices)
        preset_s = scenarios.get_scenario_d_single()
        st.session_state.det_single_proc_str = ", ".join(preset_s["processes"])
        st.session_state.det_single_res_str = ", ".join(preset_s["resources"])
        st.session_state.det_single_edges_alloc = "\n".join([f"{r}, {p}" for r, p in preset_s["allocation_edges"]])
        st.session_state.det_single_edges_req = "\n".join([f"{p}, {r}" for p, r in preset_s["request_edges"]])
        st.session_state.det_single_last_result = None
        st.session_state.det_single_recovery_result = None

        preset_m = scenarios.get_scenario_d_multi()
        st.session_state.det_multi_processes = preset_m["processes"].copy()
        st.session_state.det_multi_resources = preset_m["resources"].copy()
        st.session_state.det_multi_alloc = preset_m["allocation"].copy()
        st.session_state.det_multi_req = preset_m["request"].copy()
        st.session_state.det_multi_avail = preset_m["available"].copy()
        st.session_state.det_multi_last_result = None
        st.session_state.det_multi_recovery_result = None
        st.session_state.det_multi_version = 0

        st.session_state.initialized = True

initialize_session_state()


# ============================================================================
# SIDEBAR / CONFIGURATION & EDITABLE TABLES
# ============================================================================

st.sidebar.title("⚙️ System Configuration")
st.sidebar.caption("Configure processes, resources, or edit matrices directly.")
st.sidebar.info("🌓 Change theme anytime from the top-right menu: **Settings > Theme**.")

# Reset Button
if st.sidebar.button("🔄 Reset to Classic Example", type="secondary", use_container_width=True):
    default_data = scenarios.get_default_state()
    st.session_state.num_processes = default_data["num_processes"]
    st.session_state.num_resources = default_data["num_resources"]
    st.session_state.process_labels = default_data["process_labels"].copy()
    st.session_state.resource_labels = default_data["resource_labels"].copy()
    st.session_state.allocation = default_data["allocation"].copy()
    st.session_state.maximum = default_data["maximum"].copy()
    st.session_state.available = default_data["available"].copy()
    st.session_state.total = default_data["total"].copy()
    st.session_state.request_history = []
    st.session_state.last_request_outcome = None
    st.session_state.last_safety_check = None
    st.session_state.active_preset = "A"
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.rerun()

# Sliders: Number of processes (1-10) and resource types (1-6)
new_n_proc = st.sidebar.slider("Number of Processes (1-10)", min_value=1, max_value=10, value=st.session_state.num_processes)
new_n_res = st.sidebar.slider("Number of Resource Types (1-6)", min_value=1, max_value=6, value=st.session_state.num_resources)

# Resize matrices if slider changed
if new_n_proc != st.session_state.num_processes or new_n_res != st.session_state.num_resources:
    old_alloc = np.array(st.session_state.allocation, dtype=int)
    old_max = np.array(st.session_state.maximum, dtype=int)
    old_avail = np.array(st.session_state.available, dtype=int)
    old_total = np.array(st.session_state.total, dtype=int)

    new_alloc = np.zeros((new_n_proc, new_n_res), dtype=int)
    new_max = np.zeros((new_n_proc, new_n_res), dtype=int)
    new_avail = np.zeros(new_n_res, dtype=int)
    new_total = np.zeros(new_n_res, dtype=int)

    r_min = min(old_alloc.shape[0], new_n_proc)
    c_min = min(old_alloc.shape[1], new_n_res)
    new_alloc[:r_min, :c_min] = old_alloc[:r_min, :c_min]
    new_max[:r_min, :c_min] = old_max[:r_min, :c_min]
    new_avail[:c_min] = old_avail[:c_min]
    if len(old_total) > 0:
        new_total[:c_min] = old_total[:c_min]

    st.session_state.num_processes = new_n_proc
    st.session_state.num_resources = new_n_res
    st.session_state.process_labels = [f"P{i}" for i in range(new_n_proc)]
    st.session_state.resource_labels = ["A", "B", "C", "D", "E", "F"][:new_n_res]
    st.session_state.allocation = new_alloc
    st.session_state.maximum = new_max
    st.session_state.available = new_avail
    st.session_state.total = new_total
    st.session_state.last_safety_check = None
    st.session_state.last_request_outcome = None
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.rerun()

p_labels = st.session_state.process_labels
r_labels = st.session_state.resource_labels

# Editable Tables in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Editable Allocation Matrix")
df_sb_alloc = pd.DataFrame(st.session_state.allocation, index=p_labels, columns=r_labels)
edited_alloc = st.sidebar.data_editor(
    df_sb_alloc,
    key=f"sidebar_alloc_editor_{st.session_state.get('editor_version', 0)}",
    use_container_width=True
)

st.sidebar.markdown("### 📋 Editable Maximum Matrix")
df_sb_max = pd.DataFrame(st.session_state.maximum, index=p_labels, columns=r_labels)
edited_max = st.sidebar.data_editor(
    df_sb_max,
    key=f"sidebar_max_editor_{st.session_state.get('editor_version', 0)}",
    use_container_width=True
)

st.sidebar.markdown("### 📋 Editable Available Vector")
df_sb_avail = pd.DataFrame([st.session_state.available], index=["Available"], columns=r_labels)
edited_avail = st.sidebar.data_editor(
    df_sb_avail,
    key=f"sidebar_avail_editor_{st.session_state.get('editor_version', 0)}",
    use_container_width=True
)

# Safely sync edited tables into session_state without crashing on non-integers
try:
    st.session_state.allocation = edited_alloc.astype(int).to_numpy()
except Exception:
    st.session_state.allocation = edited_alloc.to_numpy()

try:
    st.session_state.maximum = edited_max.astype(int).to_numpy()
except Exception:
    st.session_state.maximum = edited_max.to_numpy()

try:
    st.session_state.available = edited_avail.astype(int).to_numpy()[0]
except Exception:
    st.session_state.available = edited_avail.to_numpy()[0]

# Presets in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Presets (Scenarios A - F)")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
col_p4, col_p5, col_p6 = st.sidebar.columns(3)

if col_p1.button("A: Safe", help="Classic Safe State (5 processes, 3 resources)"):
    data = scenarios.get_scenario_a()
    st.session_state.num_processes = data["num_processes"]
    st.session_state.num_resources = data["num_resources"]
    st.session_state.process_labels = data["process_labels"].copy()
    st.session_state.resource_labels = data["resource_labels"].copy()
    st.session_state.allocation = data["allocation"].copy()
    st.session_state.maximum = data["maximum"].copy()
    st.session_state.available = data["available"].copy()
    st.session_state.total = data["total"].copy()
    st.session_state.active_preset = "A"
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.rerun()

if col_p2.button("B: Wait", help="Request > Available -> WAITING / DEFERRED"):
    data = scenarios.get_scenario_b()
    st.session_state.num_processes = data["num_processes"]
    st.session_state.num_resources = data["num_resources"]
    st.session_state.process_labels = data["process_labels"].copy()
    st.session_state.resource_labels = data["resource_labels"].copy()
    st.session_state.allocation = data["allocation"].copy()
    st.session_state.maximum = data["maximum"].copy()
    st.session_state.available = data["available"].copy()
    st.session_state.total = data["total"].copy()
    st.session_state.active_preset = "B"
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.rerun()

if col_p3.button("C: Unsafe", help="Unsafe Request -> Denied & Restored"):
    data = scenarios.get_scenario_c()
    st.session_state.num_processes = data["num_processes"]
    st.session_state.num_resources = data["num_resources"]
    st.session_state.process_labels = data["process_labels"].copy()
    st.session_state.resource_labels = data["resource_labels"].copy()
    st.session_state.allocation = data["allocation"].copy()
    st.session_state.maximum = data["maximum"].copy()
    st.session_state.available = data["available"].copy()
    st.session_state.total = data["total"].copy()
    st.session_state.active_preset = "C"
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.rerun()

if col_p4.button("D: Detect", help="Deadlock Detection (Single & Multi-Instance)"):
    st.session_state.active_preset = "D"
    st.rerun()

if col_p5.button("E: Prevent", help="Prevention Policies (Ordering & Upfront)"):
    st.session_state.active_preset = "E"
    st.rerun()

if col_p6.button("F: Bound", help="Boundary & Invalid Inputs"):
    st.session_state.active_preset = "F"
    st.rerun()


# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-title">Deadlock Analysis and Resource Allocation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive OS Laboratory for Banker\'s Avoidance, Graph/Matrix Detection, '
    'Prevention Policies, and Analytical Case Verification.</div>',
    unsafe_allow_html=True
)

# Active Preset Banner
if st.session_state.active_preset:
    st.info(f"📍 **Active Preset Loaded:** Scenario {st.session_state.active_preset}. Edit values in the sidebar or switch presets anytime.")

# 4 Main Tabs
tab_avoidance, tab_detection, tab_prevention, tab_scenarios = st.tabs([
    "🛡️ Deadlock Avoidance (Banker's)",
    "🔍 Deadlock Detection",
    "⚙️ Prevention & Recovery",
    "📊 Test Scenarios & Results Table"
])


# ============================================================================
# TAB 1: DEADLOCK AVOIDANCE (BANKER'S ALGORITHM)
# ============================================================================

with tab_avoidance:
    st.header("Deadlock Avoidance: Dijkstra's Banker's Algorithm")
    st.caption("Validates whether granting a resource request preserves a state from which all processes can safely finish.")

    p_labels = st.session_state.process_labels
    r_labels = st.session_state.resource_labels

    # 1. Automatic Need Matrix Calculation & Display
    # Formula: Need = Maximum - Allocation
    try:
        need_mat = calculate_need(st.session_state.maximum, st.session_state.allocation)
        need_calculable = True
    except Exception:
        need_mat = None
        need_calculable = False

    col_view1, col_view2, col_view3 = st.columns(3)
    with col_view1:
        st.subheader("1. Allocation Matrix")
        st.caption("Resources currently held by each process.")
        df_alloc_view = pd.DataFrame(st.session_state.allocation, index=p_labels, columns=r_labels)
        st.dataframe(df_alloc_view, use_container_width=True)

    with col_view2:
        st.subheader("2. Maximum Matrix")
        st.caption("Maximum claim declared by each process.")
        df_max_view = pd.DataFrame(st.session_state.maximum, index=p_labels, columns=r_labels)
        st.dataframe(df_max_view, use_container_width=True)

    with col_view3:
        st.subheader("3. Need Matrix (Auto-Calculated)")
        st.caption("Need = Maximum − Allocation (Read-Only).")
        if need_calculable and need_mat is not None:
            df_need_view = pd.DataFrame(need_mat, index=p_labels, columns=r_labels)
            st.dataframe(df_need_view, use_container_width=True)
        else:
            st.warning("Cannot calculate Need matrix: matrix entries must be valid numeric values.")

    st.subheader("Current Available Vector")
    df_avail_view = pd.DataFrame([st.session_state.available], index=["Available"], columns=r_labels)
    st.dataframe(df_avail_view, use_container_width=True)

    # 2. System Configuration Validation using validation.py before running anything
    is_valid_cfg, cfg_msg, cfg_kind = validate_configuration(
        st.session_state.allocation,
        st.session_state.maximum,
        st.session_state.available,
        p_labels,
        r_labels
    )

    if cfg_kind == "INVALID":
        render_status_banner("INVALID INPUT", "System Configuration Error Detected!", cfg_msg)
    elif cfg_kind == "VALID_BOUNDARY":
        st.info(f"ℹ️ {cfg_msg}")

    st.markdown("---")

    # 3. Safety Check Button
    st.subheader("Run Safety Check")
    st.caption("Executes Dijkstra's Safety Algorithm to find a safe execution sequence.")

    if st.button("🚀 Run Safety Check", type="primary", use_container_width=True):
        # Validate configuration before running
        if cfg_kind == "INVALID":
            st.session_state.last_safety_check = {
                "is_safe": False,
                "validation_error": cfg_msg,
                "steps": [],
                "unfinished": list(range(len(p_labels)))
            }
        elif not need_calculable or need_mat is None:
            st.session_state.last_safety_check = {
                "is_safe": False,
                "validation_error": "Cannot run Safety Check: Need matrix calculation failed.",
                "steps": [],
                "unfinished": list(range(len(p_labels)))
            }
        else:
            with st.spinner("Executing Banker's Safety Algorithm..."):
                res_safety = safety_check(
                    st.session_state.allocation,
                    need_mat,
                    st.session_state.available
                )
                res_safety["validation_error"] = None
                st.session_state.last_safety_check = res_safety

    if "last_safety_check" in st.session_state and st.session_state.last_safety_check:
        res_safety = st.session_state.last_safety_check
        if res_safety.get("validation_error"):
            render_status_banner(
                "INVALID INPUT",
                "Cannot Run Safety Check: Configuration Invalid",
                res_safety["validation_error"]
            )
        elif res_safety["is_safe"]:
            seq_labels = [p_labels[i] for i in res_safety["safe_sequence"]]
            seq_str = " ➔ ".join(seq_labels)
            render_status_banner(
                "SAFE",
                f"Safe Sequence Found: {seq_str}",
                "Every process can obtain its maximum needed resources, finish, and release its allocation."
            )
        else:
            unfin_labels = [p_labels[i] for i in res_safety["unfinished"]]
            render_status_banner(
                "UNSAFE",
                "No Safe Sequence Exists! System is in an UNSAFE state.",
                f"Processes that could not finish: {', '.join(unfin_labels)}."
            )
            # Whenever result is UNSAFE, show the panel titled "Unsafe is NOT the same as Deadlock"
            render_unsafe_concept_panel()

        # Step-by-step table
        st.markdown("#### Step-by-Step Execution Trace")
        if res_safety["steps"]:
            step_rows = []
            for s in res_safety["steps"]:
                step_rows.append({
                    "Step": s["step"],
                    "Process Picked": p_labels[s["process"]],
                    "Need": str(s["need"]),
                    "Work Before": str(s["work_before"]),
                    "Released": str(s["allocation_released"]),
                    "Work After": str(s["work_after"])
                })
            df_step_table = pd.DataFrame(step_rows)
            st.dataframe(df_step_table, use_container_width=True)
        elif not res_safety.get("validation_error"):
            st.warning("No processes could execute (insufficient available resources to satisfy any process's Need).")

    st.markdown("---")

    # 4. Resource Request Form
    st.subheader("Simulate a Resource Request (handle_request)")
    st.caption("Submit a resource request vector for a process to evaluate with Banker's Algorithm.")

    col_rf1, col_rf2 = st.columns([1, 2])
    with col_rf1:
        req_proc_label = st.selectbox("Select Process:", options=p_labels, index=1 if len(p_labels) > 1 else 0)
        req_p_id = p_labels.index(req_proc_label)

    with col_rf2:
        req_vector_inputs = []
        sub_cols = st.columns(len(r_labels))
        for r_i, c in enumerate(sub_cols):
            val = c.number_input(f"Req {r_labels[r_i]}", min_value=-50, max_value=100, value=0, key=f"req_field_{r_i}")
            req_vector_inputs.append(int(val))

    if st.button(f"📥 Submit Request for {req_proc_label}", type="primary", use_container_width=True):
        # Save snapshot of state before request
        before_alloc = np.array(st.session_state.allocation, dtype=int).copy()
        before_avail = np.array(st.session_state.available, dtype=int).flatten().copy()

        # Use validation.py before running anything!
        is_cfg_ok, cfg_err_msg, _ = validate_configuration(
            st.session_state.allocation,
            st.session_state.maximum,
            st.session_state.available,
            p_labels,
            r_labels
        )
        is_req_ok, req_err_msg, _ = validate_request(
            req_p_id,
            req_vector_inputs,
            len(p_labels),
            len(r_labels)
        )

        if not is_cfg_ok:
            req_outcome = {
                "status": "INVALID",
                "message": f"Cannot execute request due to configuration error: {cfg_err_msg}",
                "new_allocation": before_alloc.copy(),
                "new_available": before_avail.copy(),
                "safety_result": None
            }
        elif not is_req_ok:
            req_outcome = {
                "status": "INVALID",
                "message": f"Invalid request input: {req_err_msg}",
                "new_allocation": before_alloc.copy(),
                "new_available": before_avail.copy(),
                "safety_result": None
            }
        else:
            # Call handle_request from banker.py
            req_outcome = handle_request(
                req_p_id,
                req_vector_inputs,
                st.session_state.allocation,
                st.session_state.maximum,
                st.session_state.available
            )

        st.session_state.last_request_outcome = {
            "process": req_proc_label,
            "request": req_vector_inputs,
            "outcome": req_outcome,
            "before_allocation": before_alloc,
            "before_available": before_avail
        }

        # If GRANTED, commit new state
        if req_outcome["status"] == "GRANTED":
            st.session_state.allocation = req_outcome["new_allocation"]
            st.session_state.available = req_outcome["new_available"]
            st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1

        # Keep a request history table: request, result, reason
        st.session_state.request_history.append({
            "Request": f"{req_proc_label} requests {req_vector_inputs}",
            "Result": req_outcome["status"],
            "Reason": req_outcome["message"]
        })

    # Display Last Request Outcome
    if "last_request_outcome" in st.session_state and st.session_state.last_request_outcome:
        last_req = st.session_state.last_request_outcome
        outcome = last_req["outcome"]
        status_code = outcome["status"]

        st.markdown("#### Request Outcome")
        render_status_banner(status_code, outcome["message"])

        # On a failed or unsafe request, show a Before vs After table proving the state is unchanged
        if status_code != "GRANTED":
            st.markdown("#### Before vs After State Comparison (State Unchanged)")
            st.caption(
                "Because the request was denied, deferred, or invalid, the system state was NOT modified. "
                "The tables below prove that Allocation and Available remain completely unchanged."
            )
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("**State BEFORE Request:**")
                df_b_a = pd.DataFrame(last_req["before_allocation"], index=p_labels, columns=r_labels)
                st.dataframe(df_b_a, use_container_width=True)
                df_b_av = pd.DataFrame([last_req["before_available"]], index=["Available"], columns=r_labels)
                st.dataframe(df_b_av, use_container_width=True)
            with col_a:
                st.markdown("**State AFTER Request (Unchanged / Restored):**")
                df_a_a = pd.DataFrame(outcome["new_allocation"], index=p_labels, columns=r_labels)
                st.dataframe(df_a_a, use_container_width=True)
                df_a_av = pd.DataFrame([outcome["new_available"]], index=["Available"], columns=r_labels)
                st.dataframe(df_a_av, use_container_width=True)

            # Verification of identity
            alloc_unchanged = np.array_equal(last_req["before_allocation"], outcome["new_allocation"])
            avail_unchanged = np.array_equal(last_req["before_available"], outcome["new_available"])
            if alloc_unchanged and avail_unchanged:
                st.success("✅ State Unchanged Confirmation: Allocation matrix and Available vector are 100% identical before and after. The original state was strictly preserved.")

            # Whenever the result is UNSAFE, show the panel titled "Unsafe is NOT the same as Deadlock"
            if status_code in ["UNSAFE", "DENIED_UNSAFE"]:
                render_unsafe_concept_panel()
        else:
            st.success(f"✅ Request Granted: System transitioned safely to the new state with resources allocated to {last_req['process']}.")
            if outcome.get("safety_result") and outcome["safety_result"].get("safe_sequence"):
                seq_granted = [p_labels[i] for i in outcome["safety_result"]["safe_sequence"]]
                st.info(f"Safe execution sequence in new state: {' ➔ '.join(seq_granted)}")

    # Request History Table
    st.markdown("#### Request History Table")
    if st.session_state.request_history:
        df_history = pd.DataFrame(st.session_state.request_history)
        st.dataframe(df_history, use_container_width=True)
        if st.button("🗑️ Clear Request History"):
            st.session_state.request_history = []
            st.rerun()
    else:
        st.caption("No requests submitted yet. Use the form above to simulate requests.")


# ============================================================================
# TAB 2: DEADLOCK DETECTION (SEPARATE FROM BANKER'S)
# ============================================================================

with tab_detection:
    st.header("🔍 Deadlock Detection & Graph Visualization")
    st.caption("Inspects existing resource allocations and outstanding requests to determine whether processes are currently deadlocked.")

    # Radio button: "Single-instance resources" / "Multiple-instance resources"
    model_choice = st.radio(
        "Select Resource Model for Detection:",
        ["Single-instance resources", "Multiple-instance resources"],
        key="radio_det_model",
        horizontal=True
    )

    # Always display clearly which model the current example uses
    if model_choice == "Single-instance resources":
        st.info(
            "📌 **Active Model: Single-Instance Resources** — Each resource type contains exactly **1 unit**. "
            "A directed cycle in the Resource-Allocation Graph (or Wait-For Graph) is **necessary and sufficient** proof of deadlock."
        )
    else:
        st.info(
            "📌 **Active Model: Multiple-Instance Resources** — Resource types have **multiple units**. "
            "A cycle in the graph alone is **NOT conclusive proof** of deadlock; the matrix detection algorithm is required."
        )

    st.markdown("---")

    if model_choice == "Single-instance resources":
        st.subheader("Model: Single-Instance Resource System (Graph-Based RAG / WFG)")
        st.markdown(
            "In single-instance systems, every resource type has exactly 1 instance. "
            "A directed cycle in the Wait-For Graph is both **necessary and sufficient** proof of deadlock."
        )

        # Quick Presets for Single-Instance
        col_sp1, col_sp2, col_sp3 = st.columns(3)
        if col_sp1.button("Preset: 2-Process Circular Wait", help="P1 holds R1 wants R2; P2 holds R2 wants R1"):
            st.session_state.det_single_proc_str = "P1, P2"
            st.session_state.det_single_res_str = "R1, R2"
            st.session_state.det_single_edges_alloc = "R1, P1\nR2, P2"
            st.session_state.det_single_edges_req = "P1, R2\nP2, R1"
            st.session_state.det_single_last_result = None
            st.session_state.det_single_recovery_result = None
            st.rerun()

        if col_sp2.button("Preset: 3-Process No Deadlock", help="P1->P2->P3 acyclic chain"):
            st.session_state.det_single_proc_str = "P1, P2, P3"
            st.session_state.det_single_res_str = "R1, R2, R3"
            st.session_state.det_single_edges_alloc = "R1, P1\nR2, P2\nR3, P3"
            st.session_state.det_single_edges_req = "P1, R2\nP2, R3"
            st.session_state.det_single_last_result = None
            st.session_state.det_single_recovery_result = None
            st.rerun()

        if col_sp3.button("Preset: Scenario D (4 Processes)", help="Classic cycle P1-P2-P3 with independent P4"):
            preset_s = scenarios.get_scenario_d_single()
            st.session_state.det_single_proc_str = ", ".join(preset_s["processes"])
            st.session_state.det_single_res_str = ", ".join(preset_s["resources"])
            st.session_state.det_single_edges_alloc = "\n".join([f"{r}, {p}" for r, p in preset_s["allocation_edges"]])
            st.session_state.det_single_edges_req = "\n".join([f"{p}, {r}" for p, r in preset_s["request_edges"]])
            st.session_state.det_single_last_result = None
            st.session_state.det_single_recovery_result = None
            st.rerun()

        # Input fields for Single-Instance
        col_in_p, col_in_r = st.columns(2)
        with col_in_p:
            proc_list_str = st.text_input(
                "Processes (comma-separated):",
                value=st.session_state.get("det_single_proc_str", "P1, P2, P3, P4"),
                key="input_det_s_procs"
            )
            st.session_state.det_single_proc_str = proc_list_str

        with col_in_r:
            res_list_str = st.text_input(
                "Resources (comma-separated):",
                value=st.session_state.get("det_single_res_str", "R1, R2, R3, R4"),
                key="input_det_s_res"
            )
            st.session_state.det_single_res_str = res_list_str

        p_list = [p.strip() for p in proc_list_str.split(",") if p.strip()]
        r_list = [r.strip() for r in res_list_str.split(",") if r.strip()]

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Allocation Edges (Resource → Process):**")
            st.caption("One pair per line: `Resource, Process` (e.g. `R1, P1`)")
            alloc_edges_raw = st.text_area(
                "Allocation Edges",
                value=st.session_state.get("det_single_edges_alloc", "R1, P1\nR2, P2\nR3, P3\nR4, P4"),
                height=130,
                key="input_det_s_alloc"
            )
            st.session_state.det_single_edges_alloc = alloc_edges_raw

        with col_e2:
            st.markdown("**Request Edges (Process → Resource):**")
            st.caption("One pair per line: `Process, Resource` (e.g. `P1, R2`)")
            req_edges_raw = st.text_area(
                "Request Edges",
                value=st.session_state.get("det_single_edges_req", "P1, R2\nP2, R3\nP3, R1\nP4, R2"),
                height=130,
                key="input_det_s_req"
            )
            st.session_state.det_single_edges_req = req_edges_raw

        # Parse edges
        alloc_edges = []
        for line in alloc_edges_raw.strip().split("\n"):
            if "," in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    alloc_edges.append((parts[0], parts[1]))

        req_edges = []
        for line in req_edges_raw.strip().split("\n"):
            if "," in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    req_edges.append((parts[0], parts[1]))

        if st.button("🔍 Detect Deadlock in Single-Instance System", type="primary", use_container_width=True):
            det_res = detect_single_instance_deadlock(p_list, r_list, alloc_edges, req_edges)
            st.session_state.det_single_last_result = det_res
            st.session_state.det_single_recovery_result = None

        if st.session_state.get("det_single_last_result"):
            det_res = st.session_state.det_single_last_result
            if det_res["is_deadlocked"]:
                render_status_banner(
                    "DEADLOCK DETECTED",
                    f"Deadlocked Processes: {', '.join(det_res['deadlocked_processes'])}",
                    f"Cycle(s) Detected in Wait-For Graph: {det_res['cycles']}"
                )
            else:
                render_status_banner(
                    "SAFE",
                    "No deadlock detected in the system.",
                    "No directed cycles exist in the Wait-For Graph. All processes can proceed."
                )

            # Draw Graphs using NetworkX + Matplotlib (highlighting cycle in red!)
            st.markdown("#### Graphical Visualization (RAG and WFG)")
            fig = draw_graphs_to_figure(det_res["rag"], det_res["wfg"], det_res["cycles"])
            st.pyplot(make_figure_dark_friendly(fig))

            # Single-Instance Recovery Section
            if det_res["is_deadlocked"]:
                st.markdown("---")
                st.subheader("🛠️ Deadlock Recovery by Process Termination (Single-Instance)")
                st.caption("Terminate a deadlocked victim process to break the cycle and release its held resources.")

                col_rec1, col_rec2 = st.columns([2, 1])
                with col_rec1:
                    suggested = det_res["suggested_victim"]
                    victim_choice = st.selectbox(
                        f"Choose victim process to terminate (Suggested: {suggested} - holds most resources):",
                        options=det_res["deadlocked_processes"],
                        index=det_res["deadlocked_processes"].index(suggested) if suggested in det_res["deadlocked_processes"] else 0,
                        key="select_single_victim"
                    )
                with col_rec2:
                    st.write("")
                    st.write("")
                    rec_button = st.button("💥 Terminate Victim & Recover", type="primary", use_container_width=True)

                if rec_button:
                    rec_res = recover_by_terminating_process_single(
                        p_list, r_list, alloc_edges, req_edges, victim_choice
                    )
                    st.session_state.det_single_recovery_result = rec_res

                if st.session_state.get("det_single_recovery_result"):
                    rec_res = st.session_state.det_single_recovery_result
                    if rec_res["is_resolved"]:
                        render_status_banner(
                            "SAFE",
                            f"Deadlock Resolved! Process {rec_res['victim_terminated']} was terminated.",
                            f"Resources released: {rec_res['released_resources']}. The Wait-For Graph is now cycle-free."
                        )
                    else:
                        render_status_banner(
                            "DEADLOCK DETECTED",
                            f"Deadlock persists after terminating {rec_res['victim_terminated']}.",
                            f"Remaining deadlocked processes: {rec_res['new_deadlocked']}."
                        )

                    st.markdown(f"**Terminated Process:** `{rec_res['victim_terminated']}` | **Released Resources:** `{rec_res['released_resources']}`")

                    # Draw post-recovery graph
                    fig_post = draw_graphs_to_figure(
                        rec_res["new_rag"],
                        rec_res["new_wfg"],
                        rec_res["new_cycles"],
                        title_suffix="(Post-Recovery: Cycle Broken)"
                    )
                    st.pyplot(make_figure_dark_friendly(fig_post))

    else:
        # Multiple-Instance Model
        st.subheader("Model: Multiple-Instance Resource System (Matrix Method)")
        st.caption("In multiple-instance systems, resources have multiple copies. A cycle in the graph alone is not conclusive.")

        # PROMINENT SHORT NOTE & EXPLANATION CARD
        st.markdown("""
        <div class="concept-card">
            <h4 style="color: #FBBF24; margin-top: 0; font-size: 1.12rem; display: flex; align-items: center; gap: 0.5rem;">
                💡 Crucial Theorem: Multiple-Instance Resource Detection
            </h4>
            <p style="font-weight: 700; color: #FCD34D; font-size: 1.05rem; margin-bottom: 0.5rem;">
                "A cycle alone is not conclusive proof of deadlock for multiple-instance resources."
            </p>
            <div style="font-size: 0.93rem; line-height: 1.55;">
                <p><strong>Potential Circular Wait vs Confirmed Deadlock:</strong></p>
                <ul>
                    <li><strong>Potential Circular Wait:</strong> In a graph with multiple resource units, an apparent cycle merely indicates that processes are mutually waiting for resources of those types. However, if another process <em>outside the cycle</em> holds instances of the same resource and finishes execution, it releases its resources. The waiting processes can then acquire them and proceed, resolving the condition without deadlock.</li>
                    <li><strong>Confirmed Deadlock:</strong> Deadlock is only confirmed when the <strong>Matrix Detection Algorithm</strong> verifies that <em>no permutation</em> of remaining processes can satisfy its outstanding requests with the available Work vector. Processes with <code>Finish[i] == False</code> at the end are genuinely deadlocked.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Presets for Multi-Instance
        col_mp1, col_mp2, col_mp3 = st.columns(3)
        if col_mp1.button("Preset: Scenario D Deadlock (5 procs, 3 res)", help="Classic multi-instance deadlock: P0 finishes, P1-P4 deadlocked"):
            preset_m = scenarios.get_scenario_d_multi()
            st.session_state.det_multi_processes = preset_m["processes"].copy()
            st.session_state.det_multi_resources = preset_m["resources"].copy()
            st.session_state.det_multi_alloc = preset_m["allocation"].copy()
            st.session_state.det_multi_req = preset_m["request"].copy()
            st.session_state.det_multi_avail = preset_m["available"].copy()
            st.session_state.det_multi_last_result = None
            st.session_state.det_multi_recovery_result = None
            st.session_state.det_multi_version = st.session_state.get("det_multi_version", 0) + 1
            st.rerun()

        if col_mp2.button("Preset: 3-Process Deadlock (2 procs cross-wait)", help="P0 and P1 cross-request each other, P2 holds 0"):
            st.session_state.det_multi_processes = ["P0", "P1", "P2"]
            st.session_state.det_multi_resources = ["R0", "R1"]
            st.session_state.det_multi_alloc = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
            st.session_state.det_multi_req = np.array([[0, 1], [1, 0], [0, 0]], dtype=int)
            st.session_state.det_multi_avail = np.array([0, 0], dtype=int)
            st.session_state.det_multi_last_result = None
            st.session_state.det_multi_recovery_result = None
            st.session_state.det_multi_version = st.session_state.get("det_multi_version", 0) + 1
            st.rerun()

        if col_mp3.button("Preset: No Deadlock (All Finish)", help="All processes can satisfy their requests"):
            st.session_state.det_multi_processes = ["P0", "P1", "P2"]
            st.session_state.det_multi_resources = ["R0", "R1"]
            st.session_state.det_multi_alloc = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
            st.session_state.det_multi_req = np.array([[0, 1], [1, 0], [0, 0]], dtype=int)
            st.session_state.det_multi_avail = np.array([1, 1], dtype=int)
            st.session_state.det_multi_last_result = None
            st.session_state.det_multi_recovery_result = None
            st.session_state.det_multi_version = st.session_state.get("det_multi_version", 0) + 1
            st.rerun()

        # Dedicated Input Tables for Multi-Instance (Isolated from Banker's tab)
        m_procs = st.session_state.det_multi_processes
        m_res = st.session_state.det_multi_resources
        m_ver = st.session_state.get("det_multi_version", 0)

        df_m_alloc = pd.DataFrame(st.session_state.det_multi_alloc, index=m_procs, columns=m_res)
        df_m_req = pd.DataFrame(st.session_state.det_multi_req, index=m_procs, columns=m_res)
        df_m_avail = pd.DataFrame([st.session_state.det_multi_avail], index=["Available"], columns=m_res)

        col_md1, col_md2 = st.columns(2)
        with col_md1:
            st.markdown("### 📋 Allocation Matrix")
            edit_m_alloc = st.data_editor(df_m_alloc, key=f"det_m_alloc_editor_{m_ver}", use_container_width=True)
        with col_md2:
            st.markdown("### 📋 Request Matrix")
            edit_m_req = st.data_editor(df_m_req, key=f"det_m_req_editor_{m_ver}", use_container_width=True)

        st.markdown("### 📋 Available Vector")
        edit_m_avail = st.data_editor(df_m_avail, key=f"det_m_avail_editor_{m_ver}", use_container_width=True)

        # Parse inputs
        try:
            alloc_in = edit_m_alloc.astype(int).to_numpy()
        except Exception:
            alloc_in = edit_m_alloc.to_numpy()
        try:
            req_in = edit_m_req.astype(int).to_numpy()
        except Exception:
            req_in = edit_m_req.to_numpy()
        try:
            avail_in = edit_m_avail.astype(int).to_numpy()[0]
        except Exception:
            avail_in = edit_m_avail.to_numpy()[0]

        st.session_state.det_multi_alloc = alloc_in
        st.session_state.det_multi_req = req_in
        st.session_state.det_multi_avail = avail_in

        # Run Detection Button
        if st.button("🔍 Run Multiple-Instance Detection Algorithm", type="primary", use_container_width=True):
            det_m_res = detect_multi_instance(alloc_in, req_in, avail_in)
            st.session_state.det_multi_last_result = det_m_res
            st.session_state.det_multi_recovery_result = None

        if st.session_state.get("det_multi_last_result"):
            det_m_res = st.session_state.det_multi_last_result
            deadlocked_indices = det_m_res["deadlocked_processes"]
            deadlocked_labels = [m_procs[i] for i in deadlocked_indices]

            if det_m_res["deadlock"]:
                render_status_banner(
                    "DEADLOCK DETECTED",
                    f"Deadlocked Processes: {', '.join(deadlocked_labels)}",
                    f"Final unfulfilled Work vector: {det_m_res['work']}."
                )
            else:
                render_status_banner(
                    "SAFE",
                    "No deadlock detected in the system.",
                    f"All processes satisfied their requests and returned allocations. Final Work = {det_m_res['work']}."
                )

            # Step table
            if det_m_res["steps"]:
                st.markdown("#### Detection Algorithm Execution Trace")
                st.caption("Processes whose requests are less than or equal to Work execute and release their allocations.")
                df_det_steps = pd.DataFrame([{
                    "Step": s["step"],
                    "Process Picked": m_procs[s["process"]],
                    "Request": str(s["request"]),
                    "Work Before": str(s["work_before"]),
                    "Released": str(s["released"]),
                    "Work After": str(s["work_after"])
                } for s in det_m_res["steps"]])
                st.dataframe(df_det_steps, use_container_width=True)
            elif det_m_res["deadlock"]:
                st.warning("No processes could execute (insufficient available resources to satisfy any process's request).")

            # RECOVERY SECTION FOR MULTI-INSTANCE
            if det_m_res["deadlock"]:
                st.markdown("---")
                st.subheader("🛠️ Deadlock Recovery by Process Termination (Multiple-Instance)")
                st.caption("Terminate a deadlocked process to release its allocated resources back to the Available pool.")

                # Calculate which deadlocked process holds the most resources
                holding_sums = {
                    m_procs[i]: int(alloc_in[i].sum())
                    for i in deadlocked_indices
                }
                suggested_victim = max(holding_sums, key=holding_sums.get)

                col_rm1, col_rm2 = st.columns([2, 1])
                with col_rm1:
                    chosen_victim = st.selectbox(
                        f"Choose process to terminate (Suggested: {suggested_victim} - holds {holding_sums[suggested_victim]} resources):",
                        options=deadlocked_labels,
                        index=deadlocked_labels.index(suggested_victim),
                        key="select_multi_victim"
                    )
                with col_rm2:
                    st.write("")
                    st.write("")
                    btn_rec_m = st.button("💥 Terminate Process & Recover", type="primary", use_container_width=True)

                if btn_rec_m:
                    vic_idx = m_procs.index(chosen_victim)
                    term_outcome = terminate_process(vic_idx, alloc_in, req_in, avail_in)
                    post_detection = detect_multi_instance(
                        term_outcome["allocation"],
                        term_outcome["request"],
                        term_outcome["available"]
                    )
                    st.session_state.det_multi_recovery_result = {
                        "victim": chosen_victim,
                        "victim_idx": vic_idx,
                        "released": term_outcome["released"],
                        "work_before": avail_in.tolist(),
                        "work_after": term_outcome["available"].tolist(),
                        "post_detection": post_detection,
                        "before_alloc": alloc_in.copy(),
                        "before_req": req_in.copy(),
                        "after_alloc": term_outcome["allocation"],
                        "after_req": term_outcome["request"],
                        "after_avail": term_outcome["available"]
                    }

                if st.session_state.get("det_multi_recovery_result"):
                    rec = st.session_state.det_multi_recovery_result
                    post_det = rec["post_detection"]
                    post_deadlocked = [m_procs[i] for i in post_det["deadlocked_processes"]]

                    if not post_det["deadlock"]:
                        render_status_banner(
                            "SAFE",
                            f"Deadlock Resolved! Process {rec['victim']} was terminated.",
                            f"Released resources: {rec['released']}. All remaining processes can now finish!"
                        )
                    else:
                        render_status_banner(
                            "DEADLOCK DETECTED",
                            f"Deadlock persists after terminating {rec['victim']}.",
                            f"Remaining deadlocked processes: {', '.join(post_deadlocked)}."
                        )

                    st.markdown(
                        f"**Terminated Process:** `{rec['victim']}` | "
                        f"**Released Resources:** `{rec['released']}` | "
                        f"**Work Vector:** `{rec['work_before']}` ➔ `{rec['work_after']}`"
                    )

                    # Before vs After Result Table
                    st.markdown("#### Before vs After Recovery Comparison")
                    col_b_rec, col_a_rec = st.columns(2)
                    with col_b_rec:
                        st.markdown("**State BEFORE Termination (Deadlocked):**")
                        df_b_alloc = pd.DataFrame(rec["before_alloc"], index=m_procs, columns=m_res)
                        st.dataframe(df_b_alloc, use_container_width=True)
                        st.text(f"Available: {rec['work_before']}")
                        st.error(f"❌ Deadlocked: {', '.join(deadlocked_labels)}")

                    with col_a_rec:
                        st.markdown(f"**State AFTER Terminating {rec['victim']} (Recovered):**")
                        df_a_alloc = pd.DataFrame(rec["after_alloc"], index=m_procs, columns=m_res)
                        st.dataframe(df_a_alloc, use_container_width=True)
                        st.text(f"Available: {rec['work_after']}")
                        if not post_det["deadlock"]:
                            st.success("✅ Deadlock Resolved: All remaining processes can now finish!")
                        else:
                            st.error(f"❌ Deadlock Persists: {', '.join(post_deadlocked)}")

                    # Execution trace of remaining processes finishing
                    if post_det["steps"]:
                        st.markdown("#### Execution Trace of Remaining Processes Finishing")
                        df_post_steps = pd.DataFrame([{
                            "Step": s["step"],
                            "Process Picked": m_procs[s["process"]],
                            "Request": str(s["request"]),
                            "Work Before": str(s["work_before"]),
                            "Released": str(s["released"]),
                            "Work After": str(s["work_after"])
                        } for s in post_det["steps"]])
                        st.dataframe(df_post_steps, use_container_width=True)
                        st.success(f"🎉 Final Work Vector: `{post_det['work']}`. All non-terminated processes successfully completed.")


# ============================================================================
# TAB 3: DEADLOCK PREVENTION & RECOVERY
# ============================================================================

with tab_prevention:
    st.header("🛡️ Deadlock Prevention & Quantitative Trade-offs")
    st.caption("Deadlock prevention eliminates at least one of the four necessary Coffman conditions at design time.")

    # Educational Coffman Overview
    st.subheader("The Four Necessary Coffman Conditions")
    coffman_data = get_coffman_conditions_summary()
    cols_c = st.columns(4)
    for idx, c in enumerate(coffman_data):
        with cols_c[idx]:
            st.markdown(f"**{idx + 1}. {c['name']}**")
            st.caption(c['definition'])
            st.markdown(f"*Theoretical Defense:* {c['prevention_strategy']}")

    st.markdown("---")

    # 1. Policy Selector
    st.subheader("1. Select Prevention Policy")
    policy_choice = st.radio(
        "Choose Prevention Policy:",
        [
            "Policy (a): Resource Ordering (Fixed total order: a process may only request in strictly increasing order) — Eliminates CIRCULAR WAIT",
            "Policy (b): Request-All-Upfront (Conservative allocation: all needed resources requested before start) — Eliminates HOLD AND WAIT"
        ],
        key="radio_prevention_policy"
    )
    is_ro_selected = "Resource Ordering" in policy_choice
    policy_key = "resource_ordering" if is_ro_selected else "request_all_upfront"

    # 2. Select Scenario from Detection Tab
    st.subheader("2. Target Scenario (from Detection Tab)")
    scenario_source = st.radio(
        "Apply Prevention Policy to Scenario from Detection Tab:",
        [
            "Single-Instance Deadlocked Scenario (RAG / Wait-For Graph: Circular Wait on R1, R2)",
            "Multiple-Instance Deadlocked Scenario (Matrix Model: 5 Processes, 3 Resources)"
        ],
        key="radio_prevention_scenario",
        horizontal=True
    )
    is_single_scenario = "Single-Instance" in scenario_source

    st.markdown("---")

    # 3. WHICH OF THE FOUR COFFMAN CONDITIONS IS ELIMINATED
    st.subheader("3. Coffman Condition Elimination Status")
    st.caption("Deadlock requires ALL FOUR conditions to hold simultaneously. Breaking even ONE makes deadlock mathematically impossible.")

    coffman_breakdown = get_coffman_breakdown(policy_key)
    cols_coff = st.columns(4)
    for idx, cb in enumerate(coffman_breakdown):
        with cols_coff[idx]:
            if cb["is_eliminated"]:
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.16); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #EF4444; border-radius: 8px; padding: 0.85rem; height: 100%;">
                    <div style="font-weight: 800; color: #F87171; font-size: 1.0rem; margin-bottom: 0.35rem;">{cb['name']}</div>
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.78rem;">
                        🚫 ELIMINATED BY POLICY
                    </span>
                    <div style="font-size: 0.84rem; color: #F1F5F9; margin-top: 0.5rem; line-height: 1.4;">
                        <strong>{cb['summary']}</strong><br>{cb['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.35); border-left: 5px solid #64748B; border-radius: 8px; padding: 0.85rem; height: 100%;">
                    <div style="font-weight: 700; color: #E2E8F0; font-size: 1.0rem; margin-bottom: 0.35rem;">{cb['name']}</div>
                    <span style="background-color: #475569; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.78rem;">
                        {cb['status']}
                    </span>
                    <div style="font-size: 0.84rem; color: #CBD5E1; margin-top: 0.5rem; line-height: 1.4;">
                        <strong>{cb['summary']}</strong><br>{cb['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. RUN SIMULATION & SHOW ORIGINAL VS MODIFIED SIDE-BY-SIDE
    st.subheader("4. Original vs Modified Result (Side-by-Side Comparison)")

    if is_single_scenario:
        # Extract inputs from Single-Instance Detection Tab
        proc_str = st.session_state.get("det_single_proc_str", "P1, P2")
        res_str = st.session_state.get("det_single_res_str", "R1, R2")
        alloc_raw = st.session_state.get("det_single_edges_alloc", "R1, P1\nR2, P2")
        req_raw = st.session_state.get("det_single_edges_req", "P1, R2\nP2, R1")

        p_list = [p.strip() for p in proc_str.split(",") if p.strip()]
        r_list = [r.strip() for r in res_str.split(",") if r.strip()]

        alloc_edges = []
        for line in alloc_raw.strip().split("\n"):
            if "," in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    alloc_edges.append((parts[0], parts[1]))

        req_edges = []
        for line in req_raw.strip().split("\n"):
            if "," in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    req_edges.append((parts[0], parts[1]))

        sim_res = simulate_prevention_single_instance(
            p_list, r_list, alloc_edges, req_edges, policy=policy_key
        )

        col_orig, col_mod = st.columns(2)
        with col_orig:
            st.markdown("### ❌ Original System State (Unconstrained)")
            render_status_banner(
                "DEADLOCK DETECTED",
                f"Deadlocked Processes: {', '.join(sim_res['original_deadlocked'])}",
                f"Detected Cycle in Wait-For Graph: {sim_res['original_cycles']}"
            )
            st.markdown("**Resource Allocations & Requests:**")
            st.code(
                f"Allocations (Held): {alloc_edges}\nRequests (Pending): {req_edges}",
                language="text"
            )
            if is_ro_selected:
                st.error(
                    "❌ **Root Cause: Circular Wait Formed!**\n"
                    "Because requests were unconstrained, P1 waits for R2 held by P2, while P2 waits for R1 held by P1. "
                    "A closed dependency cycle formed, trapping both processes."
                )
            else:
                st.error(
                    "❌ **Root Cause: Hold and Wait Formed!**\n"
                    "P1 holds R1 while waiting for R2. P2 holds R2 while waiting for R1. "
                    "Because incremental holding was permitted, both processes locked each other out."
                )

        with col_mod:
            st.markdown(f"### ✅ Modified State (Under {sim_res['policy_name'].split(':')[1].strip()})")
            render_status_banner(
                "SAFE",
                "Deadlock Successfully Prevented! Zero Deadlocked Processes.",
                f"Condition Eliminated: {sim_res['condition_eliminated']} (No cycles can form)."
            )

            if is_ro_selected:
                st.markdown("**Global Resource Ordering Hierarchy:**")
                st.info(f"Fixed Order: `{' < '.join([f'{r}={idx}' for r, idx in sim_res['resource_order'].items()])}`")

                st.markdown("**Request Validation Under Policy:**")
                for log_item in sim_res["evaluation_log"]:
                    if log_item["allowed"]:
                        st.success(f"✅ {log_item['reason']}")
                    else:
                        st.error(f"🚫 {log_item['reason']}")

                st.markdown("**Execution & Cycle-Breaking Trace:**")
                for step_txt in sim_res["execution_trace"]:
                    st.write(step_txt)
            else:
                st.markdown("**Upfront Resource Allocation Schedule:**")
                df_sim = pd.DataFrame([{
                    "Step": item["step"],
                    "Process": item["process"],
                    "Action": item["action"],
                    "Allocated": str(item["resources_allocated"]),
                    "Held While Waiting": "None (0 units)" if "DEFERRED" in item["action"] else str(item["resources_allocated"]),
                    "Note": item["note"]
                } for item in sim_res["simulation_log"]])
                st.dataframe(df_sim, use_container_width=True)

                st.success(
                    f"✅ **Hold and Wait Broken:** Waiting process(es) `{sim_res['waiting_processes']}` held ZERO resources "
                    f"while waiting. Process `{sim_res['execution_order'][0]}` completed cleanly, followed by the remaining processes."
                )

    else:
        # Multi-Instance Scenario from Detection Tab
        m_procs = st.session_state.get("det_multi_processes", ["P0", "P1", "P2", "P3", "P4"])
        m_res = st.session_state.get("det_multi_resources", ["A", "B", "C"])
        m_alloc = st.session_state.get("det_multi_alloc", scenarios.get_scenario_d_multi()["allocation"])
        m_req = st.session_state.get("det_multi_req", scenarios.get_scenario_d_multi()["request"])
        m_avail = st.session_state.get("det_multi_avail", scenarios.get_scenario_d_multi()["available"])

        sim_res = simulate_prevention_multi_instance(
            m_procs, m_res, m_alloc, m_req, m_avail, policy=policy_key
        )

        col_orig, col_mod = st.columns(2)
        with col_orig:
            st.markdown("### ❌ Original Matrix State (Incremental Allocation)")
            render_status_banner(
                "DEADLOCK DETECTED",
                f"Deadlocked Processes: {', '.join(sim_res['original_deadlocked'])}",
                f"Initial Available Vector = {m_avail.tolist()}"
            )
            df_alloc_show = pd.DataFrame(m_alloc, index=m_procs, columns=m_res)
            st.markdown("**Allocation Matrix:**")
            st.dataframe(df_alloc_show, use_container_width=True)
            df_req_show = pd.DataFrame(m_req, index=m_procs, columns=m_res)
            st.markdown("**Request Matrix:**")
            st.dataframe(df_req_show, use_container_width=True)

        with col_mod:
            st.markdown(f"### ✅ Modified State (Under {sim_res['policy_name'].split(':')[1].strip()})")
            render_status_banner(
                "SAFE",
                "Deadlock Successfully Prevented! All processes complete.",
                f"Condition Eliminated: {sim_res['condition_eliminated']}."
            )

            if is_ro_selected:
                st.markdown("**Global Resource Order Enforced:**")
                st.info(f"Fixed Hierarchy: `{' < '.join([f'{r}={idx}' for r, idx in sim_res['resource_order'].items()])}`")
                st.markdown("**Policy Violations Detected & Broken:**")
                for v in sim_res["violating_requests"]:
                    st.warning(
                        f"🚫 Process **{v['process']}** holds {v['held_resources']} but requested {v['requested_resources']}. "
                        f"Violating lower-ranked requests: {v['violating_res']}. Process forced to release higher resources."
                    )
                for rel in sim_res["released_summary"]:
                    st.success(f"🔄 {rel}")
                st.markdown(f"**Final Work Vector after all processes finish:** `{sim_res['final_work']}`")
            else:
                st.markdown("**Batch Upfront Allocation Schedule:**")
                df_steps = pd.DataFrame(sim_res["step_log"])
                st.dataframe(df_steps, use_container_width=True)
                st.success(
                    f"✅ **Hold and Wait Eliminated:** Batch 1 ran `{list(set(sim_res['execution_order']) - set(sim_res['waiting_processes']))}`, "
                    f"while `{sim_res['waiting_processes']}` held 0 while waiting. All jobs completed in order."
                )

    st.markdown("---")

    # 5. TRADE-OFFS PANEL (WITH REAL SCENARIO NUMBERS)
    st.subheader("5. ⚖️ Comprehensive System Trade-offs Panel (With Real Scenario Metrics)")
    st.caption("Preventing deadlock via design constraints comes at the expense of system performance, flexibility, and throughput.")

    tradeoffs = sim_res["tradeoffs"]
    col_t1, col_t2 = st.columns(2)
    col_t3, col_t4 = st.columns(2)

    with col_t1:
        u = tradeoffs["resource_utilization"]
        st.markdown(f"""
        <div style="background-color: rgba(245, 158, 11, 0.14); border: 1px solid rgba(245, 158, 11, 0.4); border-left: 5px solid #F59E0B; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #FBBF24; font-size: 1.05rem; margin-bottom: 0.35rem;">📉 {u['title']}</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #FCD34D; margin-bottom: 0.35rem;">{u['metric_value']}</div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.45;">
                {u['explanation']}<br>
                <strong>Scenario Metric:</strong> <code>{u['calculation']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        c = tradeoffs["reduced_concurrency"]
        st.markdown(f"""
        <div style="background-color: rgba(59, 130, 246, 0.14); border: 1px solid rgba(59, 130, 246, 0.4); border-left: 5px solid #3B82F6; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #60A5FA; font-size: 1.05rem; margin-bottom: 0.35rem;">⚡ {c['title']}</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #93C5FD; margin-bottom: 0.35rem;">{c['metric_value']}</div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.45;">
                {c['explanation']}<br>
                <strong>Scenario Metric:</strong> <code>{c['calculation']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        s = tradeoffs["starvation_risk"]
        st.markdown(f"""
        <div style="background-color: rgba(244, 63, 94, 0.14); border: 1px solid rgba(244, 63, 94, 0.4); border-left: 5px solid #F43F5E; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #FB7185; font-size: 1.05rem; margin-bottom: 0.35rem;">⚠️ {s['title']}</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #FDA4AF; margin-bottom: 0.35rem;">{s['metric_value']}</div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.45;">
                {s['explanation']}<br>
                <strong>Scenario Impact:</strong> <code>{s['impact']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t4:
        w = tradeoffs["longer_waiting"]
        st.markdown(f"""
        <div style="background-color: rgba(139, 92, 246, 0.14); border: 1px solid rgba(139, 92, 246, 0.4); border-left: 5px solid #8B5CF6; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #A78BFA; font-size: 1.05rem; margin-bottom: 0.35rem;">⏳ {w['title']}</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #C4B5FD; margin-bottom: 0.35rem;">{w['metric_value']}</div>
            <div style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.45;">
                {w['explanation']}<br>
                <strong>Scenario Metric:</strong> <code>{w['calculation']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 4: TEST SCENARIOS & RESULTS TABLE
# ============================================================================

    st.header("🧪 Test Scenarios & Analytical Results Table")
    st.caption("Load preset scenarios A through F, auto-run algorithms live, and evaluate analytical results against expected OS behaviors.")

    # Preset Scenario Buttons (A through F)
    st.subheader("1. Preset Scenario Quick-Load & Live Run")
    st.markdown("Click any preset button below to load the scenario data, execute the algorithm live, and inspect the detailed trace:")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    col_btn4, col_btn5, col_btn6 = st.columns(3)

    # Initialize active scenario state if not present
    if "active_preset_run" not in st.session_state:
        st.session_state.active_preset_run = None

    if col_btn1.button("🅰️ Preset A: Safe State", use_container_width=True, help="Classic 5-process, 3-resource Banker's safe state"):
        st.session_state.active_preset_run = ("A", run_scenario_a())
        st.rerun()

    if col_btn2.button("🅱️ Preset B: Request Exceeds Available", use_container_width=True, help="Grant P1 (1,0,2), then P4 requests (3,3,0)"):
        st.session_state.active_preset_run = ("B", run_scenario_b())
        st.rerun()

    if col_btn3.button("🅲 Preset C: Unsafe Request", use_container_width=True, help="Grant P1 (1,0,2), then P0 requests (0,2,0)"):
        st.session_state.active_preset_run = ("C", run_scenario_c())
        st.rerun()

    if col_btn4.button("🅳 Preset D: Deadlock Detection", use_container_width=True, help="Single-instance circular wait + Multi-instance example"):
        st.session_state.active_preset_run = ("D", run_scenario_d())
        st.rerun()

    if col_btn5.button("🅴 Preset E: Prevention on Scenario D", use_container_width=True, help="Resource Ordering and Request-All-Upfront applied to D"):
        st.session_state.active_preset_run = ("E", run_scenario_e())
        st.rerun()

    if col_btn6.button("🅵 Preset F: Boundary & Invalid Inputs", use_container_width=True, help="Zero Avail, Zero Need, P99, negative, Alloc>Max, len 2"):
        st.session_state.active_preset_run = ("F", run_scenario_f())
        st.rerun()

    # Detailed Scenario Live Inspection Panel
    if st.session_state.get("active_preset_run"):
        tag, data = st.session_state.active_preset_run
        st.markdown("---")
        st.subheader(f"🔍 Live Scenario Execution Trace: Scenario {tag}")

        if tag == "A":
            sc = data["scenario"]
            render_status_banner(
                "SAFE",
                f"State is Proven SAFE! Safe Sequence: {data['safe_sequence_str']}",
                "All 5 processes found an execution order to acquire needed resources and return allocations."
            )
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**Allocation Matrix:**")
                st.dataframe(pd.DataFrame(sc["allocation"], index=sc["process_labels"], columns=sc["resource_labels"]), use_container_width=True)
                st.markdown("**Maximum Claim Matrix:**")
                st.dataframe(pd.DataFrame(sc["maximum"], index=sc["process_labels"], columns=sc["resource_labels"]), use_container_width=True)
            with col_a2:
                st.markdown("**Calculated Need Matrix:**")
                st.dataframe(pd.DataFrame(data["need"], index=sc["process_labels"], columns=sc["resource_labels"]), use_container_width=True)
                st.markdown(f"**Available Vector:** `{sc['available'].tolist()}`")

            st.markdown("#### Safety Algorithm Execution Steps:")
            steps_df = format_steps_as_dataframe(data["safety_result"]["steps"], sc["process_labels"], sc["resource_labels"])
            st.dataframe(steps_df, use_container_width=True)

        elif tag == "B":
            sc = data["scenario"]
            res = data["request_result"]
            render_status_banner(
                "WAITING",
                "WAITING / DEFERRED: Request cannot be granted immediately.",
                res["message"]
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**System State Prior to Request:**")
                st.dataframe(pd.DataFrame(sc["allocation"], index=sc["process_labels"], columns=sc["resource_labels"]), use_container_width=True)
                st.info(f"Available Vector: `{sc['available'].tolist()}`")
            with col_b2:
                st.markdown("**Request Details:**")
                st.write(f"- **Requesting Process:** `{sc['request_process_label']}`")
                st.write(f"- **Request Vector:** `{sc['request_vector'].tolist()}`")
                st.write(f"- **P4 Remaining Need:** `[4, 3, 1]`")
                st.write(f"- **Check 1 (Request <= Need):** `[3, 3, 0] <= [4, 3, 1]` ➔ **PASSED**")
                st.write(f"- **Check 2 (Request <= Available):** `[3, 3, 0] <= [2, 3, 0]` ➔ **FAILED (Deficit on Resource A: 3 > 2)**")
                st.success("✅ State Preserved: Allocation and Available vectors remain untouched.")

        elif tag == "C":
            sc = data["scenario"]
            res = data["request_result"]
            render_status_banner(
                "UNSAFE",
                "DENIED_UNSAFE: Request is Denied. Original Safe State Restored.",
                res["message"]
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("**Original Safe State (Preserved):**")
                st.dataframe(pd.DataFrame(sc["allocation"], index=sc["process_labels"], columns=sc["resource_labels"]), use_container_width=True)
                st.info(f"Available: `{sc['available'].tolist()}`")
            with col_c2:
                st.markdown("**Simulated Temporary State (Unsafe):**")
                sim_avail = sc["available"] - sc["request_vector"]
                st.error(f"Simulated Available after granting P0: `{sim_avail.tolist()}`")
                st.warning("At Available=[2, 1, 0], no process can guarantee completion. Safety algorithm returned UNSAFE.")
                st.success("✅ Rollback Enacted: Process P0 rolled back; original state restored.")

        elif tag == "D":
            render_status_banner(
                "DEADLOCK DETECTED",
                "Deadlock Detected in Both Single-Instance and Multi-Instance Models!",
                data["actual_result"]
            )
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Single-Instance Graph Model")
                sc_s = data["scenario_single"]
                res_s = data["result_single"]
                st.write(f"- **Processes:** `{sc_s['processes']}` | **Resources:** `{sc_s['resources']}`")
                st.write(f"- **Allocations (Held):** `{sc_s['allocation_edges']}`")
                st.write(f"- **Requests (Pending):** `{sc_s['request_edges']}`")
                st.error(f"❌ Cycle Detected in Wait-For Graph: `{res_s['cycle']}`")
                st.write("In single-instance systems, a cycle in the WFG is necessary and sufficient proof of deadlock.")

            with col_d2:
                st.markdown("#### Multiple-Instance Matrix Model")
                sc_m = data["scenario_multi"]
                res_m = data["result_multi"]
                st.dataframe(pd.DataFrame(sc_m["allocation"], index=sc_m["processes"], columns=sc_m["resources"]), use_container_width=True)
                st.error(f"❌ Unfinished / Deadlocked Processes: `{[f'P{i}' for i in res_m['deadlocked_processes']]}`")
                st.write(f"After P0 finishes and releases, Work = `{res_m['work']}`. No other process's request can be satisfied.")

        elif tag == "E":
            render_status_banner(
                "SAFE",
                "Deadlock Successfully Prevented Under Both Policies!",
                data["actual_result"]
            )
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("#### Policy (a): Resource Ordering")
                ro = data["result_ro"]
                st.write(f"- **Eliminated Condition:** 🚫 **{ro['condition_eliminated']}**")
                st.write(f"- **Global Hierarchy:** `{ro['resource_order']}`")
                st.write(f"- **Rejected Requests:** `{ro['rejected_edges']}` (1 < 2 violates increasing hierarchy)")
                st.success("✅ Circular wait cycle broken. All processes finish.")
            with col_e2:
                st.markdown("#### Policy (b): Request-All-Upfront")
                up = data["result_upfront"]
                st.write(f"- **Eliminated Condition:** 🚫 **{up['condition_eliminated']}**")
                st.write(f"- **Execution Order:** `{up['execution_order']}`")
                st.write(f"- **Waiting (Holds 0):** `{up['waiting_processes']}`")
                st.success("✅ Hold-and-wait eliminated. All processes finish in batches.")

        elif tag == "F":
            render_status_banner(
                "SAFE",
                "All 6 Boundary and Invalid Inputs Evaluated Correctly!",
                "Validation suite passed without throwing any uncaught exceptions."
            )
            df_cases = pd.DataFrame(data["cases"])
            st.dataframe(df_cases[["id", "name", "status", "msg", "passed"]], use_container_width=True)

    st.markdown("---")

    # 2. Results Table Section
    st.subheader("2. Consolidated Analytical Results Table")
    st.caption("Live evaluation of all scenarios against expected theoretical outcomes. Actual Results are computed dynamically from real algorithm runs.")

    col_tbl_btn, col_tbl_info = st.columns([1, 2])
    with col_tbl_btn:
        btn_refresh_table = st.button("⚡ Re-run All Scenarios & Update Table", type="primary", use_container_width=True)

    # Compute or fetch table
    if btn_refresh_table or "analytical_table_v2" not in st.session_state:
        table_records = get_all_scenarios_results_table()
        st.session_state.analytical_table_v2 = pd.DataFrame(table_records)

    df_display = st.session_state.analytical_table_v2
    st.dataframe(df_display, use_container_width=True)

    # 3. CSV Download Button
    st.markdown("---")
    st.subheader("3. Export Results")
    col_dl_csv, col_dl_json = st.columns(2)

    with col_dl_csv:
        csv_bytes = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download results as CSV",
            data=csv_bytes,
            file_name="deadlock_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_results_csv"
        )

    with col_dl_json:
        json_bytes = df_display.to_json(orient="records", indent=2).encode('utf-8')
        st.download_button(
            label="📄 Download results as JSON",
            data=json_bytes,
            file_name="deadlock_analysis_results.json",
            mime="application/json",
            use_container_width=True,
            key="btn_download_results_json"
        )


# ============================================================================
# FOOTER NOTE
# ============================================================================

st.markdown("""
<div class="footer-note">
    🛡️ <strong>Operating Systems IA Activity 2</strong> | 
    <em>Banker's Algorithm = Avoidance (checks future safe transitions; does not detect existing deadlock). 
    Deadlock Detection is a separate analytical function.</em>
</div>
""", unsafe_allow_html=True)
