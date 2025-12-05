import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# -------------------- CONFIGURATION --------------------
st.set_page_config(
    page_title="OptiFlow | Industrial Scheduler",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS & THEME (DARK MODE) --------------------
st.markdown("""
<style>
    /* 1. Main Background */
    .stApp {
        background-color: #0f172a; /* Dark Blue/Slate Background */
        color: #e2e8f0; /* Light text */
    }
    
    /* 2. Headers */
    h1, h2, h3, h4, .css-xq1adw-StMarkdown {
        color: #f8f9fa; /* Off-White Headers */
        font-family: 'Segoe UI', sans-serif;
    }

    /* 3. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Slightly lighter dark blue for contrast */
        border-right: 1px solid #334155;
    }

    /* 4. Custom Card Style (Background must be light enough to show details) */
    .metric-card {
        background-color: #1e293b; /* Dark card background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
        border-top: 4px solid #3b82f6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff; /* White text for values */
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8; /* Grayish text for labels */
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 5. General Text/Widget Backgrounds (e.g., input boxes, data editor) */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input, .stDataFrame, [data-testid="stTextarea"], .st-bc, .st-bb {
        background-color: #334155 !important;
        color: #f8f9fa !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SCHEDULING ALGORITHMS & SIMULATION --------------------

def simulate_flowshop(p, sequence, m, n, r=None, s=None):
    """
    Simulates a Flow Shop schedule and calculates completion times.
    p: matrix of processing times (n x m)
    sequence: list of job indices (0-based)
    m: number of machines
    n: number of jobs
    r: release dates (currently unused in UI but kept for robustness)
    s: setup times (currently unused in UI but kept for robustness)
    Returns: makespan, Completion time matrix (C)
    """
    if r is None: r = np.zeros(n)
    C = np.zeros((n, m))
    
    for pos in range(n):
        job = sequence[pos]
        prev_job = sequence[pos-1] if pos > 0 else None
        
        for mach in range(m):
            release_time = r[job] if mach == 0 else 0
            prev_job_finish = C[pos-1, mach] if pos > 0 else 0
            prev_mach_finish = C[pos, mach-1] if mach > 0 else 0
            
            # Setup time is assumed zero for now as manual setup input is not fully implemented
            setup_time = 0
            # if s is not None and prev_job is not None:
            #     setup_time = s[prev_job, job, mach]
            
            # Machine 1 (mach=0) start time: max(previous job finished on M1, job release date) + setup
            if mach == 0:
                start_time = max(prev_job_finish, release_time) + setup_time
            # Other machines start time: max(previous machine finished this job, previous job finished on this machine) + setup
            else:
                start_time = max(prev_mach_finish, prev_job_finish) + setup_time
            
            C[pos, mach] = start_time + p[job, mach]
            
    return C[-1, -1], C

def johnson_rule(p, m, n, r=None, s=None):
    """Johnson's rule for 2 machines."""
    A, B = p[:,0], p[:,1]
    set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
    idx1, idx2 = set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]
    sequence = np.concatenate([idx1, idx2]).tolist()
    ms, C = simulate_flowshop(p, sequence, m, n, r, s)
    return sequence, ms, C

def heuristic_schedule(rule, p, m, n, r, d=None, s=None):
    """Applies a common scheduling heuristic (SPT, LPT, EDD)."""
    if rule == 'EDD': 
        idx = np.argsort(d)
    elif rule == 'WDD': 
        idx = np.argsort(-d) # WDD is not explicitly calculated, using -d as placeholder for Lateness
    elif rule == 'SPT': 
        idx = np.argsort(np.sum(p, axis=1))
    elif rule == 'LPT': 
        idx = np.argsort(-np.sum(p, axis=1))
    else:
        raise ValueError("Unsupported Heuristic")
        
    sequence = idx.tolist()
    ms, C = simulate_flowshop(p, sequence, m, n, r, s)
    return sequence, ms, C

def cds_heuristic(p, m, n, r=None, s=None):
    """Campbell, Dudek & Smith heuristic for m > 2 machines."""
    num_k = m - 1
    best_ms, best_seq, best_C = np.inf, None, None
    for k in range(1, num_k+1):
        # Create 2-machine problem proxies
        A, B = np.sum(p[:, :k], axis=1), np.sum(p[:, m-k:m], axis=1)
        
        # Apply Johnson's Rule to the proxy problem
        set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
        seq = np.concatenate([set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]]).tolist()
        
        # Simulate the original m-machine problem with the new sequence
        ms, C_temp = simulate_flowshop(p, seq, m, n, r, s)
        
        # Track the best sequence found
        if ms < best_ms:
            best_ms, best_seq, best_C = ms, seq, C_temp
    return best_seq, best_ms, best_C

def compute_metrics(C, r, d, n, m):
    """Calculates performance metrics (TFT, TT, TAR, TFR)."""
    Cm = C[:, m-1] # Completion times on the last machine
    
    # Flow Time (F_j = C_j - r_j)
    F = Cm - r
    TFT = np.sum(F)
    TFR = TFT / n
    
    # Tardiness (T_j = max(0, C_j - d_j))
    if np.all(d <= 0):
        TT = 0.0
        TAR = 0.0
    else:
        Tard = np.maximum(0, Cm - d)
        TT = np.sum(Tard)
        TAR = TT / n
        
    return TFT, TT, TAR, TFR

# -------------------- VISUALIZATION (MATPLOTLIB) --------------------
def plot_gantt_matplotlib(seq, C, p_mat, ms, m, n, title):
    """Classic Gantt Chart using Matplotlib - Clean & Clear"""
    import matplotlib.pyplot as plt
    
    seq0 = [s for s in seq]  # Already 0-based from algorithms
    
    fig, ax = plt.subplots(figsize=(14, 2.5 + 0.8*m))
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    y_spacing = 1.5
    y_positions = [(mach+1)*y_spacing for mach in range(m)]
    
    for mach in range(m):
        y = y_positions[mach]
        for pos in range(n):
            job = seq0[pos]
            dur = p_mat[job, mach]
            finish = C[pos, mach]
            start = finish - dur
            
            # Draw bar
            ax.barh(y, dur, left=start, height=0.7, 
                   edgecolor='black', linewidth=1.5, color=colors[job])
            
            # Add job label
            if dur > 0.5:
                ax.text(start + dur/2, y, f"J{job+1}", 
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='#333')
        
        # Machine label
        ax.text(-0.05*ms, y, f"Machine {mach+1}", 
               ha='right', va='center', 
               fontsize=11, fontweight='bold', color='#2563eb')
    
    # Styling
    ax.set_xlim(0, ms * 1.05)
    ax.set_ylim(0, max(y_positions)+0.8)
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_title(f"{title} (Makespan: {ms:.2f})", fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

# -------------------- UI COMPONENTS --------------------
def render_metric_card(label, value, color="#3b82f6"):
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- MAIN APP --------------------

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏭 OptiFlow")
    st.caption("Flow Shop Production Scheduler")
    st.markdown("---")
    
    # System Type
    st.subheader("1. System Configuration")
    system_type = st.selectbox("Workshop Type", ["Flow Shop (Sans Préparation)", "Flow Shop (Avec Préparation)", "Job Shop"], index=0)
    
    # Input Mode
    st.subheader("2. Data Source")
    input_mode = st.radio("Input Method", ["Manual Entry", "Excel Import", "Random Demo"], index=2)

    st.markdown("---")
    if "Flow Shop" in system_type:
        if "Sans Préparation" in system_type:
            st.info("💡 **Flow Shop (Sans Préparation):** Tous les travaux suivent la même séquence. Pas de temps de setup.\n\n"
                    "**Algorithmes appliqués:**\n"
                    "- m=2: SPT, LPT, EDD, WDD, Johnson\n"
                    "- m≥3: SPT, LPT, EDD, WDD, CDS(k=1 à m-1)")
        else:
            st.info("💡 **Flow Shop (Avec Préparation):** Tous les travaux suivent la même séquence avec temps de setup entre jobs.\n\n"
                    "**Algorithmes appliqués:**\n"
                    "- m=2: SPT, LPT, EDD, WDD, Johnson\n"
                    "- m≥3: SPT, LPT, EDD, WDD, CDS(k=1 à m-1)")
            st.warning("⚠️ **Setup Times:** Fonctionnalité en développement")
    else:
        st.warning("⚠️ **Job Shop:** Module en cours de développement. Utilisez Flow Shop.")
        
# Initialize variables in session state for cross-mode continuity
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()
if 'n' not in st.session_state:
    st.session_state.n = 0
if 'm' not in st.session_state:
    st.session_state.m = 0

# --- MAIN CONTENT ---
st.markdown("## 📊 Scheduling Dashboard")

# DATA INPUT LOGIC
if input_mode == "Manual Entry":
    st.subheader("📝 Manual Data Entry")
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Number of Jobs (n)", 2, 50, 4)
    with col2:
        m = st.number_input("Number of Machines (m)", 2, 20, 3)
    
    rows = [f"M{i+1}" for i in range(m)] + ["Due Date"]
    cols = [f"J{j+1}" for j in range(n)]
    df_init = pd.DataFrame(np.zeros((m+1, n)), index=rows, columns=cols)
    
    st.session_state.edited_df = st.data_editor(df_init, use_container_width=True, key="manual_editor")
    st.session_state.n = n
    st.session_state.m = m
    
elif input_mode == "Random Demo":
    st.subheader("🎲 Random Data Generation")
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of Jobs", 3, 20, 5)
    with col2:
        m = st.slider("Number of Machines", 2, 10, 3)
    
    np.random.seed(42)
    p_data = np.random.randint(1, 20, size=(m, n))
    d_data = np.random.randint(20, 100, size=(1, n))
    full_data = np.vstack([p_data, d_data])
    
    rows = [f"M{i+1}" for i in range(m)] + ["Due Date"]
    cols = [f"J{j+1}" for j in range(n)]
    st.session_state.edited_df = pd.DataFrame(full_data, index=rows, columns=cols)
    st.session_state.n = n
    st.session_state.m = m
    st.dataframe(st.session_state.edited_df, use_container_width=True, height=200)

elif input_mode == "Excel Import":
    st.subheader("📄 Excel Data Import")
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'], help="File must have Machine/Due Date rows and Job columns.")
    if uploaded_file:
        try:
            df_temp = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.m = len(df_temp.index) - 1
            st.session_state.n = len(df_temp.columns)
            st.session_state.edited_df = df_temp
            st.success(f"Loaded: **{st.session_state.n} Jobs** on **{st.session_state.m} Machines**")
            st.dataframe(df_temp, height=150)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Please upload a file to proceed.")
        st.stop()


# --- CALCULATION TRIGGER ---
st.markdown("---")
col_action, col_blank = st.columns([1, 4])
with col_action:
    calculate = st.button("🚀 Optimize Schedule", type="primary", use_container_width=True)

if calculate:
    m = st.session_state.m
    n = st.session_state.n
    edited_df = st.session_state.edited_df

    if "Job Shop" in system_type:
        st.error("Job Shop module is currently under maintenance. Please use Flow Shop.")
        st.stop()
        
    if "Avec Préparation" in system_type:
        st.warning("⚠️ Flow Shop avec préparation n'est pas encore complètement implémenté. Calcul sans setup times...")

    if n < 2 or m < 2:
        st.error("You need at least 2 Jobs and 2 Machines to perform scheduling.")
        st.stop()

    try:
        # Pre-process Data
        data_matrix = edited_df.values.astype(float)
        p_mat = data_matrix[:m, :].T  # (Jobs x Machines)
        d_arr = data_matrix[m, :]     # Due Dates
        r_arr = np.zeros(n)           # Release dates assumed 0
        
        # --- RUN ALGORITHMS ---
        results = []
        
        # Base heuristics for ALL cases (m >= 1)
        # 1. SPT
        seq, ms, C = heuristic_schedule('SPT', p_mat, m, n, r_arr, d_arr, None)
        results.append({"Method": "SPT", "Seq": seq, "Makespan": ms, "C": C})
        
        # 2. LPT
        seq, ms, C = heuristic_schedule('LPT', p_mat, m, n, r_arr, d_arr, None)
        results.append({"Method": "LPT", "Seq": seq, "Makespan": ms, "C": C})

        # 3. EDD
        seq, ms, C = heuristic_schedule('EDD', p_mat, m, n, r_arr, d_arr, None)
        results.append({"Method": "EDD", "Seq": seq, "Makespan": ms, "C": C})
        
        # 4. WDD
        seq, ms, C = heuristic_schedule('WDD', p_mat, m, n, r_arr, d_arr, None)
        results.append({"Method": "WDD", "Seq": seq, "Makespan": ms, "C": C})
        
        # Johnson ONLY for m=2
        if m == 2:
            seq, ms, C = johnson_rule(p_mat, m, n, r_arr, None)
            results.append({"Method": "Johnson's Rule", "Seq": seq, "Makespan": ms, "C": C})
            
        # CDS ONLY for m >= 3
        cds_all_results = []
        if m >= 3:
            num_k = m - 1
            for k in range(1, num_k + 1):
                # Create 2-machine problem proxies
                A = np.sum(p_mat[:, :k], axis=1)
                B = np.sum(p_mat[:, m-k:m], axis=1)
                
                # Apply Johnson's Rule to the proxy problem
                set1 = np.where(A < B)[0]
                set2 = np.where(B <= A)[0]
                seq = np.concatenate([set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]]).tolist()
                
                # Simulate the original m-machine problem with the new sequence
                ms, C_temp = simulate_flowshop(p_mat, seq, m, n, r_arr, None)
                
                cds_all_results.append({
                    "Method": f"CDS (k={k})",
                    "Seq": seq,
                    "Makespan": ms,
                    "C": C_temp,
                    "k": k
                })
            
            # Add all CDS results to main results
            results.extend(cds_all_results)

        # Find Best Result based on Makespan
        best_res = min(results, key=lambda x: x['Makespan'])
        
        # --- DISPLAY RESULTS ---
        
        st.markdown("### 🏆 Optimization Results")
        
        # Calculate metrics for the best result
        tft, tt, tar, tfr = compute_metrics(best_res['C'], r_arr, d_arr, n, m)
        
        # 1. Top Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1: render_metric_card("Best Makespan (Cmax)", f"{best_res['Makespan']:.1f}", "#10b981")
        with m2: render_metric_card("Total Tardiness (TT)", f"{tt:.1f}", "#ef4444")
        with m3: render_metric_card("Avg Flow Time (TFR)", f"{tfr:.1f}", "#3b82f6")
        with m4: render_metric_card("Best Method", best_res['Method'], "#8b5cf6")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Tabs for Details
        tab1, tab2, tab3 = st.tabs(["📊 Gantt Charts", "📈 Comparison Table", "💾 Export Data"])
        
        with tab1:
            st.markdown(f"#### Winning Schedule: {best_res['Method']}")
            
            # Matplotlib Gantt - Clean & Clear
            fig = plot_gantt_matplotlib(best_res['Seq'], best_res['C'], p_mat, best_res['Makespan'], m, n, best_res['Method'])
            st.pyplot(fig)
            
            # Summary under the chart
            seq_1based = [i + 1 for i in best_res['Seq']]
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.markdown(f"**📋 Séquence optimale:** J{seq_1based}")
            with col_sum2:
                st.markdown(f"**⏱️ Makespan:** {best_res['Makespan']:.2f}")
            
            st.markdown("---")
            
            with st.expander("📊 Comparer avec les autres méthodes"):
                for res in results:
                    if res != best_res:
                        sub_fig = plot_gantt_matplotlib(res['Seq'], res['C'], p_mat, res['Makespan'], m, n, res['Method'])
                        st.pyplot(sub_fig)
                        
                        # Summary for each comparison chart
                        seq_comp = [i + 1 for i in res['Seq']]
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.markdown(f"**📋 Séquence:** J{seq_comp}")
                        with col_c2:
                            st.markdown(f"**⏱️ Makespan:** {res['Makespan']:.2f}")
                        st.markdown("---")

        with tab2:
            # Create comparison DataFrame
            comp_data = []
            for res in results:
                _tft, _tt, _tar, _tfr = compute_metrics(res['C'], r_arr, d_arr, n, m)
                seq_str = " → ".join([f"J{i+1}" for i in res['Seq']])
                comp_data.append({
                    "Algorithm": res['Method'],
                    "Makespan": res['Makespan'],
                    "Total Tardiness": _tt,
                    "Avg Flow Time": _tfr,
                    "Total Flow Time": _tft,
                    "Sequence": seq_str
                })
            
            comp_df = pd.DataFrame(comp_data).sort_values(by="Makespan").reset_index(drop=True)
            
            # Format numbers to 2 decimals for display
            display_df = comp_df.copy()
            display_df['Makespan'] = display_df['Makespan'].apply(lambda x: f"{x:.2f}")
            display_df['Total Tardiness'] = display_df['Total Tardiness'].apply(lambda x: f"{x:.2f}")
            display_df['Avg Flow Time'] = display_df['Avg Flow Time'].apply(lambda x: f"{x:.2f}")
            display_df['Total Flow Time'] = display_df['Total Flow Time'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True, height=400)

        with tab3:
            st.markdown("Download the comparison results for reporting.")
            csv = comp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name='scheduling_results.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")
        st.warning("Please ensure your input data is purely numeric, non-negative, and the structure is correct.")