import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OptiFlow | Planificateur Industriel",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

st.markdown(
"""
**Réalisé par :**
- BOULOUDNINE Younes
- AYACH Ahmed
- EL ANSARI Souhail
"""
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CSS PERSONNALISÉ & THÈME (MODE SOMBRE)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* 1. Fond Principal */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
  
    /* 2. En-têtes */
    h1, h2, h3, h4, .css-xq1adw-StMarkdown {
        color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* 3. Style de la Barre Latérale */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* 4. Style de Carte Métrique */
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
        border-top: 4px solid #3b82f6;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* 5. Fonds de Texte/Widgets Généraux */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input, 
    .stDataFrame, [data-testid="stTextarea"], .st-bc, .st-bb {
        background-color: #334155 !important;
        color: #f8f9fa !important;
    }
    
    /* 6. Amélioration des tableaux */
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    
    .stDataFrame th {
        background-color: #1e293b;
        color: #f8f9fa;
        font-weight: bold;
        padding: 8px;
        text-align: center;
    }
    
    .stDataFrame td {
        background-color: #334155;
        color: #f8f9fa;
        padding: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ALGORITHMES FLOW SHOP
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_flowshop(p, sequence, m, n, r=None, s=None):
    """Simule un planning Flow Shop SANS temps de préparation."""
    if r is None: 
        r = np.zeros(n)
    
    C = np.zeros((n, m))
    
    for pos in range(n):
        job = sequence[pos]
        
        for mach in range(m):
            release_time = r[job] if mach == 0 else 0
            prev_job_finish = C[pos-1, mach] if pos > 0 else 0
            prev_mach_finish = C[pos, mach-1] if mach > 0 else 0
            
            if mach == 0:
                start_time = max(prev_job_finish, release_time)
            else:
                start_time = max(prev_mach_finish, prev_job_finish)
            
            C[pos, mach] = start_time + p[job, mach]
    
    return C[-1, -1], C

def simulate_flowshop_with_setup(p, sequence, setup_matrices, m, n, r=None):
    """Simule un planning Flow Shop AVEC temps de préparation."""
    if r is None:
        r = np.zeros(n)
    
    C = np.zeros((n, m))
    S = np.zeros((n, m))
    setup_times = np.zeros((n, m))
    
    for pos in range(n):
        job = sequence[pos]
        prev_job = sequence[pos-1] if pos > 0 else None
        
        for mach in range(m):
            if prev_job is None:
                setup_time = setup_matrices[mach][job, job]
            else:
                setup_time = setup_matrices[mach][prev_job, job]
            
            setup_times[pos, mach] = setup_time
            
            machine_free = C[pos-1, mach] if pos > 0 else 0.0
            job_arrival = r[job] if mach == 0 else C[pos, mach-1]
            start_setup = machine_free
            S[pos, mach] = start_setup
            setup_end = start_setup + setup_time
            processing_start = max(job_arrival, setup_end)
            C[pos, mach] = processing_start + p[job, mach]
    
    makespan = C[-1, -1]
    return makespan, C, S, setup_times

def johnson_rule(p, m, n, r=None, s=None):
    """Règle de Johnson pour 2 machines."""
    A, B = p[:,0], p[:,1]
    set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
    idx1, idx2 = set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]
    sequence = np.concatenate([idx1, idx2]).tolist()
    ms, C = simulate_flowshop(p, sequence, m, n, r, s)
    return sequence, ms, C

def johnson_rule_M2M1(p, m, n, r=None, s=None):
    """
    Règle de Johnson appliquée pour la séquence M2 -> M1.
    A = t_M2 (première machine), B = t_M1 (deuxième machine).
    """
    A, B = p[:, 0], p[:, 1]  # A = t_M2, B = t_M1
   
    set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
   
    # Tri de Set 1 (A croissant, c-à-d t_M2 croissant)
    idx1 = set1[np.argsort(A[set1])]
   
    # Tri de Set 2 (B décroissant, c-à-d t_M1 décroissant)
    idx2 = set2[np.argsort(-B[set2])]
   
    sequence = np.concatenate([idx1, idx2]).tolist()
    ms, C = simulate_flowshop(p, sequence, m, n, r, s)
    return sequence, ms, C

def johnson_rule_with_setup(p, setup_matrices, m, n, r=None):
    """Règle de Johnson pour 2 machines AVEC préparation."""
    A, B = p[:,0], p[:,1]
    set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
    idx1, idx2 = set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]
    sequence = np.concatenate([idx1, idx2]).tolist()
    ms, C, S, setup_t = simulate_flowshop_with_setup(p, sequence, setup_matrices, m, n, r)
    return sequence, ms, C, S, setup_t

def heuristic_schedule(rule, p, m, n, r, d=None, s=None):
    """Applique une heuristique courante (SPT, LPT, EDD, WDD)."""
    if rule == 'EDD':
        idx = np.argsort(d)
    elif rule == 'WDD':
        idx = np.argsort(-d)
    elif rule == 'SPT':
        idx = np.argsort(np.sum(p, axis=1))
    elif rule == 'LPT':
        idx = np.argsort(-np.sum(p, axis=1))
    else:
        raise ValueError("Heuristique non supportée")
    
    sequence = idx.tolist()
    ms, C = simulate_flowshop(p, sequence, m, n, r, s)
    return sequence, ms, C

def heuristic_schedule_with_setup(rule, p, setup_matrices, m, n, r, d=None):
    """Applique une heuristique courante AVEC temps de préparation."""
    if rule == 'EDD':
        idx = np.argsort(d)
    elif rule == 'WDD':
        idx = np.argsort(-d)
    elif rule == 'SPT':
        idx = np.argsort(np.sum(p, axis=1))
    elif rule == 'LPT':
        idx = np.argsort(-np.sum(p, axis=1))
    elif rule in ['TPS_asc', 'TPS_desc']:
        tps = np.zeros(n)
        for j in range(n):
            proc = np.sum(p[j])
            setup_s = sum(setup_matrices[k][j,j] for k in range(m))
            tps[j] = proc + setup_s
        if rule == 'TPS_asc':
            idx = np.argsort(tps)
        else:
            idx = np.argsort(-tps)
    else:
        raise ValueError("Heuristique non supportée")
    
    sequence = idx.tolist()
    ms, C, S, setup_t = simulate_flowshop_with_setup(p, sequence, setup_matrices, m, n, r)
    return sequence, ms, C, S, setup_t

def cds_heuristic(p, m, n, r=None, s=None):
    """Heuristique de Campbell, Dudek & Smith pour m > 2 machines."""
    num_k = m - 1
    best_ms, best_seq, best_C = np.inf, None, None
    for k in range(1, num_k+1):
        A, B = np.sum(p[:, :k], axis=1), np.sum(p[:, m-k:m], axis=1)
        set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
        seq = np.concatenate([set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]]).tolist()
        ms, C_temp = simulate_flowshop(p, seq, m, n, r, s)
        if ms < best_ms:
            best_ms, best_seq, best_C = ms, seq, C_temp
    return best_seq, best_ms, best_C

def cds_heuristic_with_setup(p, setup_matrices, m, n, r=None):
    """Heuristique de Campbell, Dudek & Smith AVEC temps de préparation."""
    num_k = m - 1
    best_ms, best_seq, best_C, best_S, best_setup = np.inf, None, None, None, None
    for k in range(1, num_k+1):
        A, B = np.sum(p[:, :k], axis=1), np.sum(p[:, m-k:m], axis=1)
        set1, set2 = np.where(A < B)[0], np.where(B <= A)[0]
        seq = np.concatenate([set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]]).tolist()
        ms, C_temp, S_temp, setup_tmp = simulate_flowshop_with_setup(p, seq, setup_matrices, m, n, r)
        if ms < best_ms:
            best_ms, best_seq, best_C, best_S, best_setup = ms, seq, C_temp, S_temp, setup_tmp
    return best_seq, best_ms, best_C, best_S, best_setup

def compute_metrics(C, r, d, n, m):
    """Calcule les métriques de performance (TFT, TT, TAR, TFR)."""
    Cm = C[:, m-1]
    F = Cm - r
    TFT = np.sum(F)
    TFR = TFT / n
    
    if np.all(d <= 0):
        TT = 0.0
        TAR = 0.0
    else:
        Tard = np.maximum(0, Cm - d)
        TT = np.sum(Tard)
        TAR = TT / n
    
    return TFT, TT, TAR, TFR

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ALGORITHMES JOB SHOP
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHME JOB SHOP CORRIGÉ - PLANIFICATION PAR NIVEAU D'OPÉRATION
# ═══════════════════════════════════════════════════════════════════════════════
class JobShopJob:
    """Représente un travail avec sa séquence de machines et temps de traitement"""
    def __init__(self, job_id, operations):
        """
        operations: liste de tuples (machine_id, processing_time)
        Exemple: [(0, 5), (2, 3), (1, 4)] signifie M1→M3→M2
        """
        self.job_id = job_id
        self.operations = operations
        self.total_processing = sum(time for _, time in operations)
        self.num_operations = len(operations)
    
    def __repr__(self):
        return f"J{self.job_id+1}: {self.operations}"


def jobshop_heuristic(jobs, rule='SPT'):
    """
    Applique SPT ou LPT au Job Shop.
    
    Étapes:
    1. Calculer le temps total de chaque job (somme sur toutes machines)
    2. Trier par temps total (croissant pour SPT, décroissant pour LPT)
    3. Créer la séquence de priorité
    4. Planifier niveau par niveau avec simulate_jobshop_priority_based
    """
    
    # Nombre de machines
    m_machines = max(max(op[0] for op in job.operations) for job in jobs) + 1
    
    # Tri stable par temps total puis job_id
    if rule == 'SPT':
        sorted_jobs = sorted(jobs, key=lambda job: (job.total_processing, job.job_id))
    elif rule == 'LPT':
        sorted_jobs = sorted(jobs, key=lambda job: (-job.total_processing, job.job_id))
    else:
        raise ValueError("Rule must be 'SPT' or 'LPT'")
    
    # Séquence de priorité (ordre des job_id)
    sequence = [job.job_id for job in sorted_jobs]
    
    # Simulation niveau par niveau
    makespan, schedule, gantt_data = simulate_jobshop_priority_based(jobs, sequence, m_machines)
    
    return sequence, makespan, schedule, gantt_data


def simulate_jobshop_priority_based(jobs, sequence, m_machines):
    """
    Simule Job Shop avec priorité globale - NIVEAU PAR NIVEAU.
    
    Algorithme:
    1. Chaque job a une priorité globale (basée sur SPT/LPT)
    2. À chaque niveau d'opération (Op1, Op2, Op3...):
       - On traite les jobs dans l'ordre de priorité de la séquence
       - Chaque job va sur SA machine spécifique pour cette opération
       - On respecte: machine libre ET fin de l'opération précédente du job
    
    Exemple avec séquence [J3, J1, J2, J4]:
        Niveau Op1: 
            - J3 sur M1 (temps 0-7)
            - J1 sur M2 (temps 0-5)
            - J2 sur M3 (temps 0-6)
            - J4 sur M2 (temps 5-9, car M2 occupée par J1 jusqu'à 5)
        
        Niveau Op2:
            - J3 sur M2 (temps max(7, 5) = 7-10, attend fin Op1 de J3 ET M2 libre)
            - J1 sur M3 (temps max(5, 6) = 6-9)
            - J2 sur M1 (temps max(6, 7) = 7-9)
            - J4 sur M1 (temps max(9, 9) = 9-14)
        
        etc.
    
    Args:
        jobs: Liste d'objets JobShopJob
        sequence: Ordre de priorité [job_id, job_id, ...]
        m_machines: Nombre de machines
    
    Returns:
        makespan, schedule, gantt_data
    """
    
    n = len(jobs)
    job_dict = {job.job_id: job for job in jobs}
    
    # Temps de disponibilité de chaque machine
    machine_available = [0.0] * m_machines
    
    # Temps de fin de dernière opération pour chaque job
    job_last_finish = {job.job_id: 0.0 for job in jobs}
    
    # Planning détaillé
    schedule = []
    gantt_data = {m: [] for m in range(m_machines)}
    
    # Trouver le nombre max d'opérations parmi tous les jobs
    max_operations = max(len(job.operations) for job in jobs)
    
    # ═══ PLANIFICATION NIVEAU PAR NIVEAU ═══
    for op_level in range(max_operations):
        print(f"\n{'='*70}")
        print(f"NIVEAU OPÉRATION {op_level + 1}")
        print(f"{'='*70}")
        
        # Pour ce niveau, traiter les jobs dans l'ORDRE DE PRIORITÉ
        for job_id in sequence:
            job = job_dict[job_id]
            
            # Vérifier si ce job a une opération à ce niveau
            if op_level >= len(job.operations):
                print(f"  J{job_id+1}: Pas d'opération à ce niveau (job terminé)")
                continue
            
            # Obtenir l'opération courante pour ce job
            machine_id, proc_time = job.operations[op_level]
            
            # Calculer le temps de démarrage
            # Le job doit attendre DEUX choses:
            # 1. Que la machine soit libre
            # 2. Que son opération précédente soit terminée
            machine_free = machine_available[machine_id]
            job_ready = job_last_finish[job_id]
            
            start_time = max(machine_free, job_ready)
            end_time = start_time + proc_time
            
            print(f"  J{job_id+1} → M{machine_id+1}: "
                  f"attend M{machine_id+1} libre ({machine_free:.1f}) "
                  f"ET fin Op précédente ({job_ready:.1f}) "
                  f"→ démarre à {start_time:.1f}, finit à {end_time:.1f}")
            
            # Mettre à jour les temps
            machine_available[machine_id] = end_time
            job_last_finish[job_id] = end_time
            
            # Enregistrer dans le planning
            schedule.append({
                'job_id': job_id,
                'machine': machine_id,
                'start': start_time,
                'end': end_time,
                'operation': op_level,
                'duration': proc_time
            })
            
            gantt_data[machine_id].append({
                'job': job_id,
                'start': start_time,
                'end': end_time,
                'operation': op_level
            })
    
    makespan = max(machine_available) if machine_available else 0.0
    return makespan, schedule, gantt_data

def jackson_algorithm(jobs, m_machines):
    """ Implémente l'algorithme de Jackson pour Job Shop à 2 machines"""
    if m_machines != 2:
        raise ValueError("Jackson algorithm only works for 2 machines")
    
    # Classer les jobs dans 4 groupes
    E1 = []  # Jobs n'utilisant que M1
    E2 = []  # Jobs n'utilisant que M2
    E12 = []  # Jobs utilisant M1 puis M2
    E21 = []  # Jobs utilisant M2 puis M1
    
    for job in jobs:
        machines_used = [op[0] for op in job.operations]
        if len(machines_used) == 1:
            if machines_used[0] == 0:
                E1.append((job.job_id, job.operations[0][1]))  # (job_id, time_M1)
            else:
                E2.append((job.job_id, job.operations[0][1]))  # (job_id, time_M2)
        elif len(machines_used) == 2:
            if machines_used == [0, 1]:  # M1 → M2
                E12.append((job.job_id, job.operations[0][1], job.operations[1][1]))
            elif machines_used == [1, 0]:  # M2 → M1
                E21.append((job.job_id, job.operations[0][1], job.operations[1][1]))
    
    # Trier E12 et E21 avec la règle de Johnson
    # Pour E12: utiliser Johnson sur (temps_M1, temps_M2)
    if E12:
        p_E12 = np.array([[t1, t2] for _, t1, t2 in E12])
        seq_E12, _, _ = johnson_rule(p_E12, 2, len(E12))
        sorted_E12 = [E12[i] for i in seq_E12]
    else:
        sorted_E12 = []
    
    # Pour E21: utiliser Johnson sur (temps_M2, temps_M1)
    if E21:
        p_E21 = np.array([[t1, t2] for _, t1, t2 in E21])  # Inverser M1 et M2
        seq_E21, _, _ = johnson_rule_M2M1(p_E21, 2, len(E21))
        sorted_E21 = [E21[i] for i in seq_E21]
    else:
        sorted_E21 = []
    
    # Construire les séquences finales
    # Machine 1: E12 → E1 → E21 (mais pour E21, c'est la 2ème opération)
    sequence_M1 = []
    # Ajouter E12 (1ère opération sur M1)
    for job_id, t1, t2 in sorted_E12:
        sequence_M1.append((job_id, t1, 'first'))
    # Ajouter E1 (tout sur M1)
    for job_id, t in E1:
        sequence_M1.append((job_id, t, 'only'))
    # Ajouter E21 (2ème opération sur M1)
    for job_id, t1, t2 in sorted_E21:  # t1 = t_M2, t2 = t_M1
        sequence_M1.append((job_id, t2, 'second'))
    
    # Machine 2: E21 → E2 → E12
    sequence_M2 = []
    # Ajouter E21 (1ère opération sur M2)
    for job_id, t1, t2 in sorted_E21:  # t1 = t_M2
        sequence_M2.append((job_id, t1, 'first'))
    # Ajouter E2 (tout sur M2)
    for job_id, t in E2:
        sequence_M2.append((job_id, t, 'only'))
    # Ajouter E12 (2ème opération sur M2)
    for job_id, t1, t2 in sorted_E12:  # t2 = t_M2
        sequence_M2.append((job_id, t2, 'second'))
    
    return sequence_M1, sequence_M2, E1, E2, sorted_E12, sorted_E21

def simulate_jobshop_jackson(sequence_M1, sequence_M2, jobs):
    """Simulation pour l'algorithme de Jackson"""
    m = 2
    n = len(jobs)
    job_dict = {job.job_id: job for job in jobs}
    machine_available = [0.0, 0.0]
    job_next_op = {job.job_id: 0 for job in jobs}
    completed_jobs = 0
    schedule = []
    gantt_data = {0: [], 1: []}
    
    # Build ordered lists of (job_id, op_idx) for each machine
    m1_ops = []
    for op in sequence_M1:
        job_id, duration, op_type = op
        if op_type == 'only':
            op_idx = 0
        elif op_type == 'first':
            op_idx = 0
        elif op_type == 'second':
            op_idx = 1
        m1_ops.append((job_id, op_idx))
    
    m2_ops = []
    for op in sequence_M2:
        job_id, duration, op_type = op
        if op_type == 'only':
            op_idx = 0
        elif op_type == 'first':
            op_idx = 0
        elif op_type == 'second':
            op_idx = 1
        m2_ops.append((job_id, op_idx))
    
    ptr_M1 = 0
    ptr_M2 = 0
    
    while completed_jobs < n:
        # Advance to next machine available time
        time = min(machine_available)
        
        # Try to schedule at current time for free machines
        scheduled_something = True
        while scheduled_something:
            scheduled_something = False
            for machine in range(m):
                if machine_available[machine] > time:
                    continue
                
                # Get next op for this machine
                if machine == 0:
                    if ptr_M1 >= len(m1_ops):
                        continue
                    job_id, op_idx = m1_ops[ptr_M1]
                    seq_ptr = ptr_M1
                else:
                    if ptr_M2 >= len(m2_ops):
                        continue
                    job_id, op_idx = m2_ops[ptr_M2]
                    seq_ptr = ptr_M2
                
                # Check if this op is next for the job
                if job_next_op.get(job_id, -1) != op_idx:
                    continue  # Not ready yet, skip to next
                
                # Ready to schedule
                job = job_dict[job_id]
                if op_idx >= len(job.operations):
                    # Invalid, skip
                    if machine == 0:
                        ptr_M1 += 1
                    else:
                        ptr_M2 += 1
                    continue
                
                machine_id, proc_time = job.operations[op_idx]
                if machine_id != machine:
                    # Mismatch, skip
                    if machine == 0:
                        ptr_M1 += 1
                    else:
                        ptr_M2 += 1
                    continue
                
                start_time = time
                end_time = start_time + proc_time
                machine_available[machine] = end_time
                
                # Record
                schedule.append({
                    'job_id': job_id,
                    'machine': machine,
                    'start': start_time,
                    'end': end_time,
                    'operation': op_idx,
                    'duration': proc_time
                })
                gantt_data[machine].append({
                    'job': job_id,
                    'start': start_time,
                    'end': end_time,
                    'operation': op_idx
                })
                
                job_next_op[job_id] = op_idx + 1
                if job_next_op[job_id] >= job.num_operations:
                    completed_jobs += 1
                
                # Advance pointer
                if machine == 0:
                    ptr_M1 += 1
                else:
                    ptr_M2 += 1
                
                scheduled_something = True
    
    makespan = max(machine_available)
    return makespan, schedule, gantt_data

# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION FLOW SHOP
# ═══════════════════════════════════════════════════════════════════════════════
def plot_gantt_matplotlib(seq, C, p_mat, ms, m, n, title):
    """Diagramme de Gantt classique avec Matplotlib - SANS Préparation"""
    fig, ax = plt.subplots(figsize=(14, 2.5 + 0.8*m))
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    y_spacing = 1.5
    y_positions = [(mach+1)*y_spacing for mach in range(m)]
    
    for mach in range(m):
        y = y_positions[mach]
        for pos in range(n):
            job = seq[pos]
            dur = p_mat[job, mach]
            finish = C[pos, mach]
            start = finish - dur
            
            ax.barh(y, dur, left=start, height=0.7,
                   edgecolor='black', linewidth=1.5, color=colors[job])
            
            if dur > 0.5:
                ax.text(start + dur/2, y, f"J{job+1}",
                       ha='center', va='center',
                       fontsize=10, fontweight='bold', color='#333')
        
        ax.text(-0.05*ms, y, f"Machine {mach+1}",
               ha='right', va='center',
               fontsize=11, fontweight='bold', color='#2563eb')
    
    ax.set_xlim(0, ms * 1.05)
    ax.set_ylim(0, max(y_positions)+0.8)
    ax.set_xlabel("Temps", fontsize=12, fontweight='bold')
    ax.set_title(f"{title} (Makespan: {ms:.2f})", fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_gantt_with_setup(seq, C, S, setup_times, p_mat, ms, m, n, title):
    """Diagramme de Gantt AVEC Temps de Préparation"""
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(16, 3 + 0.8*m))
    
    setup_color = '#FF6B6B'
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    y_spacing = 1.5
    y_positions = [(mach+1)*y_spacing for mach in range(m)]
    
    legend_elements = [Patch(facecolor=setup_color, edgecolor='black', label='Temps de Préparation')]
    for i in range(min(n, 3)):
        legend_elements.append(Patch(facecolor=colors[i], edgecolor='black', label=f'Travail {i+1}'))
    
    for mach in range(m):
        y = y_positions[mach]
        for pos in range(n):
            job = seq[pos]
            
            setup_start = S[pos, mach]
            setup_dur = setup_times[pos, mach]
            if setup_dur > 0:
                ax.barh(y, setup_dur, left=setup_start, height=0.7,
                       edgecolor='black', linewidth=1, color=setup_color, alpha=0.9)
                if setup_dur > 1:
                    ax.text(setup_start + setup_dur/2, y, f"S:{setup_dur:.0f}",
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
            
            processing_dur = p_mat[job, mach]
            processing_start = C[pos, mach] - processing_dur
            ax.barh(y, processing_dur, left=processing_start, height=0.7,
                   edgecolor='black', linewidth=1.5, color=colors[job])
            
            if processing_dur > 0.5:
                ax.text(processing_start + processing_dur/2, y, f"J{job+1}",
                       ha='center', va='center', fontsize=10, fontweight='bold', color='#333')
        
        ax.text(-0.03*ms, y, f"M{mach+1}",
               ha='right', va='center', fontsize=11, fontweight='bold', color='#2563eb')
    
    ax.set_xlim(0, ms * 1.05)
    ax.set_ylim(0, max(y_positions)+0.8)
    ax.set_xlabel("Temps", fontsize=12, fontweight='bold')
    ax.set_title(f"{title} (Makespan: {ms:.2f})", fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION JOB SHOP
# ═══════════════════════════════════════════════════════════════════════════════
def plot_jobshop_gantt(gantt_data, jobs, makespan, m_machines, title):
    """Gantt chart for Job Shop"""
    fig, ax = plt.subplots(figsize=(14, 4 + 0.6 * m_machines))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(jobs)))
    job_colors = {job.job_id: colors[idx % len(colors)] for idx, job in enumerate(jobs)}
    
    y_spacing = 1.0
    # Create y positions from BOTTOM to TOP (Machine 1 at bottom)
    y_positions = {m: (m + 1) * y_spacing for m in range(m_machines)}
    
    # Plot each machine
    for machine in range(m_machines):
        y = y_positions[machine]
        
        # Sort blocks by start time for this machine
        machine_blocks = gantt_data.get(machine, [])
        machine_blocks.sort(key=lambda x: x['start'])
        
        for block in machine_blocks:
            job_id = block['job']
            start = block['start']
            end = block['end']
            duration = end - start
            op_idx = block['operation']
            
            # Validate data
            if duration <= 0:
                continue
            
            # Draw the rectangle
            rect = Rectangle((start, y - 0.3), duration, 0.6,
                            facecolor=job_colors[job_id],
                            edgecolor='black',
                            linewidth=1,
                            alpha=0.8)
            ax.add_patch(rect)
            
            # Add text label
            if duration > 0.5:
                # Display: Job-Operation (Duration)
                label = f"J{job_id+1}-O{op_idx+1}\n({duration:.0f})"
                ax.text(start + duration/2, y, label,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white')
        
        # Machine label on left
        ax.text(-0.02 * makespan, y, f"M{machine+1}",
               ha='right', va='center',
               fontsize=11, fontweight='bold', color='#2563eb')
    
    # Set axis limits
    ax.set_xlim(0, max(makespan, 1) * 1.05)
    ax.set_ylim(0, (m_machines + 1) * y_spacing)
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_title(f"{title} (Makespan: {makespan:.1f})", fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    legend_patches = []
    for job in jobs[:min(8, len(jobs))]:
        patch = Rectangle((0, 0), 1, 1, facecolor=job_colors[job.job_id])
        legend_patches.append((patch, f"J{job.job_id+1}"))
    
    if legend_patches:
        ax.legend([p[0] for p in legend_patches],
                 [p[1] for p in legend_patches],
                 loc='upper right', bbox_to_anchor=(1.15, 1),
                 title="Jobs")
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPOSANTS UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_metric_card(label, value, color="#3b82f6"):
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BARRE LATÉRALE
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏭 OptiFlow")
    st.caption("Planificateur de Production")
    st.markdown("---")
    
    # ═══ SÉLECTION DU TYPE D'ATELIER ═══
    st.subheader("1. Type d'Atelier")
    shop_type = st.radio(
        "Sélectionnez la Configuration de l'Atelier",
        ["🔄 Flow Shop", "🔀 Job Shop"],
        index=0
    )
    
    # ═══ CONFIGURATION DU SYSTÈME ═══
    if "Flow Shop" in shop_type:
        st.subheader("2. Configuration Flow Shop")
        system_type = st.selectbox(
            "Temps de Préparation",
            ["Sans Préparation", "Avec Préparation"],
            index=0
        )
    else:
        system_type = "Sans Préparation"  # Job Shop n'utilise pas les temps de préparation dans cette version
    
    # ═══ MODE D'ENTRÉE ═══
    st.subheader("3. Source de Données")
    input_mode = st.radio("Méthode d'Entrée", ["Saisie Manuelle", "Import Excel", "Démonstration Aléatoire"], index=2)
    st.markdown("---")
    
    # ═══ BOÎTES D'INFO ═══
    if "Flow Shop" in shop_type:
        if "Sans Préparation" in system_type:
            st.info("💡 **Flow Shop (Sans Préparation):** Tous les travaux suivent la même séquence.\n\n"
                    "**Algorithmes:**\n"
                    "- m=2: SPT, LPT, EDD, WDD, Johnson\n"
                    "- m≥3: SPT, LPT, EDD, WDD, CDS")
        else:
            st.info("💡 **Flow Shop (Avec Préparation):** Avec temps de setup entre jobs.\n\n"
                    "**Algorithmes:**\n"
                    "- m=2: SPT, LPT, EDD, WDD, Johnson, TPS\n"
                    "- m≥3: SPT, LPT, EDD, WDD, CDS, TPS")
    else:
        st.info("💡 **Job Shop:** Chaque travail a sa propre séquence de machines.\n\n"
                "**Algorithmes:**\n"
                "- SPT (Shortest Processing Time)\n"
                "- LPT (Longest Processing Time)\n"
                "- Jackson (2 machines seulement)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. ÉTAT DE SESSION INITIAL
# ═══════════════════════════════════════════════════════════════════════════════
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()
if 'n' not in st.session_state:
    st.session_state.n = 0
if 'm' not in st.session_state:
    st.session_state.m = 0

# ═══ SPÉCIFIQUE JOB SHOP ═══
if 'machine_sequence_df' not in st.session_state:
    st.session_state.machine_sequence_df = pd.DataFrame()
if 'jobshop_jobs' not in st.session_state:
    st.session_state.jobshop_jobs = []

# ═══════════════════════════════════════════════════════════════════════════════
# 10. CONTENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Tableau de Bord de Planification")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. ENTRÉE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
if "Flow Shop" in shop_type:
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRÉE DONNÉES FLOW SHOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    if input_mode == "Saisie Manuelle":
        st.subheader("📝 Saisie Manuelle - Flow Shop")
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nombre de Travails (n)", 2, 50, 4)
        with col2:
            m = st.number_input("Nombre de Machines (m)", 2, 20, 3)
        
        rows = [f"M{i+1}" for i in range(m)] + ["Date Limite"]
        cols = [f"J{j+1}" for j in range(n)]
        df_init = pd.DataFrame(np.zeros((m+1, n)), index=rows, columns=cols)
        
        st.session_state.edited_df = st.data_editor(df_init, use_container_width=True, key="manual_editor")
        st.session_state.n = n
        st.session_state.m = m
        
        if "Avec Préparation" in system_type:
            st.markdown("---")
            st.subheader("⚙️ Configuration des Temps de Préparation")
            st.info(f"💡 Créez **{m} matrices {n}×{n}** (une par machine)")
            
            setup_tabs = st.tabs([f"🔧 Machine {i+1}" for i in range(m)])
            setup_matrices = []
            
            for machine_idx, tab in enumerate(setup_tabs):
                with tab:
                    st.markdown(f"### Temps de Préparation - Machine {machine_idx + 1}")
                    
                    if f'setup_m{machine_idx}' not in st.session_state:
                        np.random.seed(42 + machine_idx)
                        default_setup = np.random.randint(1, 5, size=(n, n))
                        st.session_state[f'setup_m{machine_idx}'] = default_setup
                    
                    rows = [f"De J{j+1}" for j in range(n)]
                    cols = [f"Vers J{k+1}" for k in range(n)]
                    
                    setup_df = pd.DataFrame(
                        st.session_state[f'setup_m{machine_idx}'],
                        index=rows,
                        columns=cols
                    )
                    
                    edited_setup = st.data_editor(
                        setup_df,
                        use_container_width=True,
                        key=f"setup_editor_m{machine_idx}",
                        height=300
                    )
                    
                    setup_matrices.append(edited_setup.values.astype(float))
                    
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        if st.button(f"🔄 Réinitialiser", key=f"reset_m{machine_idx}"):
                            np.random.seed(42 + machine_idx)
                            st.session_state[f'setup_m{machine_idx}'] = np.random.randint(1, 5, size=(n, n))
                            st.rerun()
                    with col_q2:
                        if st.button(f"⚖️ Tous à 2", key=f"uni_m{machine_idx}"):
                            st.session_state[f'setup_m{machine_idx}'] = np.full((n, n), 2.0)
                            st.rerun()
                    with col_q3:
                        if st.button(f"⭕ Tous à 0", key=f"zero_m{machine_idx}"):
                            st.session_state[f'setup_m{machine_idx}'] = np.zeros((n, n))
                            st.rerun()
            
            st.session_state.setup_matrices = setup_matrices
    
    elif input_mode == "Démonstration Aléatoire":
        st.subheader("🎲 Génération de Données Aléatoires - Flow Shop")
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Nombre de Travails", 3, 20, 5)
        with col2:
            m = st.slider("Nombre de Machines", 2, 10, 3)
        
        np.random.seed(42)
        p_data = np.random.randint(1, 20, size=(m, n))
        d_data = np.random.randint(20, 100, size=(1, n))
        full_data = np.vstack([p_data, d_data])
        
        rows = [f"M{i+1}" for i in range(m)] + ["Date Limite"]
        cols = [f"J{j+1}" for j in range(n)]
        st.session_state.edited_df = pd.DataFrame(full_data, index=rows, columns=cols)
        st.session_state.n = n
        st.session_state.m = m
        st.dataframe(st.session_state.edited_df, use_container_width=True, height=200)
        
        if "Avec Préparation" in system_type:
            st.markdown("---")
            st.subheader("⚙️ Matrices de Préparation Générées")
            
            setup_matrices = []
            for machine_idx in range(m):
                np.random.seed(100 + machine_idx)
                setup_matrix = np.random.randint(1, 5, size=(n, n))
                setup_matrices.append(setup_matrix)
            
            st.session_state.setup_matrices = setup_matrices
            
            with st.expander("📋 Voir les matrices de préparation"):
                for machine_idx in range(m):
                    st.markdown(f"**Machine {machine_idx+1}:**")
                    rows = [f"De J{j+1}" for j in range(n)]
                    cols = [f"Vers J{k+1}" for k in range(n)]
                    df_setup = pd.DataFrame(setup_matrices[machine_idx], index=rows, columns=cols)
                    st.dataframe(df_setup, use_container_width=True)
    
    elif input_mode == "Import Excel":
        st.subheader("📄 Import de Données Excel - Flow Shop")
        st.info("""
        **Format Excel Requis:**
        
        **Feuille Principale (ex: "Temps_Traitement"):** Temps de traitement et dates limites.
        ```
               J1 J2 J3 J4
        M1 4 6 8 6
        M2 5 7 3 5
        M3 3 2 5 3
        Due 30 25 40 35
        ```
        - Lignes: Machines M1-Mm, dernière ligne "Date Limite" ou "Due"
        - Colonnes: Travails J1-Jn
        
        **Pour Préparations (si activé):** Feuilles supplémentaires nommées "Setup_M1", "Setup_M2", ..., "Setup_Mm".
        Chaque feuille est une matrice n×n:
        ```
               Vers J1 Vers J2 Vers J3 Vers J4
        De J1 0 2 1 3
        De J2 1 0 4 2
        De J3 3 1 0 1
        De J4 2 3 2 0
        ```
        - Diagonale typiquement 0 (setup job sur lui-même).
        - Si une feuille Setup_Mi manquante: Génération aléatoire.
        """)
        
        uploaded_file = st.file_uploader("Télécharger Fichier Excel", type=['xlsx'], key="flow_upload")
        
        if uploaded_file:
            try:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # Chercher feuille principale pour temps de traitement
                main_sheet = None
                for sheet in sheet_names:
                    sheet_lower = sheet.lower()
                    if "traitement" in sheet_lower or "proc" in sheet_lower or "temps" in sheet_lower or sheet == sheet_names[0]:
                        main_sheet = sheet
                        break
                if not main_sheet:
                    main_sheet = sheet_names[0]
                
                df = pd.read_excel(excel_file, sheet_name=main_sheet, index_col=0, header=0)
                
                # Corriger noms de colonnes si numériques
                if df.columns.dtype in ['int64', 'float64']:
                    df.columns = [f"J{int(col)}" for col in df.columns]
                
                st.session_state.n = len(df.columns)
                st.session_state.m = len(df.index) - 1  # moins ligne date limite
                st.session_state.edited_df = df
                
                st.success(f"✅ Chargé: **{st.session_state.n} Travails** × **{st.session_state.m} Machines**")
                st.dataframe(df, height=200)
                
                if "Avec Préparation" in system_type:
                    setup_matrices = []
                    for machine_idx in range(st.session_state.m):
                        sheet_name = f"Setup_M{machine_idx+1}"
                        try:
                            setup_df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0, header=0)
                            # Corriger noms si nécessaire
                            if setup_df.columns.dtype in ['int64', 'float64']:
                                setup_df.columns = [f"Vers J{int(col)}" for col in setup_df.columns]
                            if setup_df.index.dtype in ['int64', 'float64']:
                                setup_df.index = [f"De J{int(idx)}" for idx in setup_df.index]
                            setup_matrices.append(setup_df.values.astype(float))
                            st.success(f"✅ Setup Machine {machine_idx+1} importé depuis '{sheet_name}'")
                            
                            with st.expander(f"📋 Aperçu Setup Machine {machine_idx+1}"):
                                rows_s = [f"De J{j+1}" for j in range(st.session_state.n)]
                                cols_s = [f"Vers J{k+1}" for k in range(st.session_state.n)]
                                df_setup_display = pd.DataFrame(setup_matrices[-1], index=rows_s, columns=cols_s)
                                st.dataframe(df_setup_display, use_container_width=True)
                        except Exception as setup_e:
                            st.warning(f"⚠️ Setup Machine {machine_idx+1} non trouvé ('{sheet_name}'). Génération aléatoire: {setup_e}")
                            np.random.seed(100 + machine_idx)
                            setup_matrix = np.random.randint(1, 5, size=(st.session_state.n, st.session_state.n))
                            setup_matrices.append(setup_matrix)
                    
                    st.session_state.setup_matrices = setup_matrices
            
            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
                import traceback
                st.code(traceback.format_exc(), language='python')

else:
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRÉE DONNÉES JOB SHOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    if input_mode == "Saisie Manuelle":
        st.subheader("📝 Saisie Manuelle - Job Shop")
        st.info("💡 **Job Shop nécessite DEUX tableaux:**\n"
                "1. **Temps de Traitement** (0 = opération non nécessaire)\n"
                "2. **Séquence de Machines** (ordre des opérations, indexé 1)")
        
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nombre de Jobs (n)", 2, 20, 4, key="js_n")
        with col2:
            m = st.number_input("Nombre de Machines (m)", 2, 10, 4, key="js_m")
        
        st.session_state.n = n
        st.session_state.m = m
        
        col_proc, col_seq = st.columns(2)
        
        with col_proc:
            st.markdown("#### 1️⃣ Temps de Traitement")
            rows = [f"M{i+1}" for i in range(m)]
            cols = [f"J{j+1}" for j in range(n)]
            df_proc = pd.DataFrame(np.zeros((m, n)), index=rows, columns=cols)
            st.session_state.edited_df = st.data_editor(df_proc, use_container_width=True, key="js_proc_editor")
        
        with col_seq:
            st.markdown("#### 2️⃣ Séquence de Machines")
            st.caption("Entrez les numéros de machines (1 à m). Utilisez 0 pour opérations inutilisées.")
            max_ops = m
            rows_seq = [f"Op{i+1}" for i in range(max_ops)]
            df_seq = pd.DataFrame(np.zeros((max_ops, n)), index=rows_seq, columns=cols)
            st.session_state.machine_sequence_df = st.data_editor(df_seq, use_container_width=True, key="js_seq_editor")
        
        # Aperçu des Itinéraires Interprétés
        if st.session_state.n > 0 and st.session_state.m > 0:
            st.markdown("---")
            st.markdown("#### 📋 Job Routes Interprétés:")
            st.caption("💡 Ceci montre comment le programme interprète vos tableaux. Modifiez ci-dessus pour mettre à jour.")
            
            proc_df = st.session_state.edited_df
            seq_df = st.session_state.machine_sequence_df
            max_ops = len(seq_df.index)
            
            for j in range(n):
                operations = []
                for op_idx in range(max_ops):
                    machine_num = seq_df.iloc[op_idx, j] if pd.notna(seq_df.iloc[op_idx, j]) else 0
                    if float(machine_num) > 0:
                        machine_idx = int(machine_num) - 1
                        if machine_idx < m:
                            proc_time = proc_df.iloc[machine_idx, j]
                            if pd.notna(proc_time) and float(proc_time) > 0:
                                operations.append((machine_idx, float(proc_time)))
                
                if operations:
                    route_str = " → ".join([f"M{m_idx+1}({t:.1f})" for m_idx, t in operations])
                    st.success(f"**Job {j+1}:** {route_str}")
                else:
                    st.warning(f"⚠️ **Job {j+1}:** Aucune opération valide. Vérifiez séquence et temps de traitement.")
    
    elif input_mode == "Démonstration Aléatoire":
        st.subheader("🎲 Génération de Données Aléatoires - Job Shop")
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Nombre de Jobs", 3, 10, 4, key="js_n_slider")
        with col2:
            m = st.slider("Nombre de Machines", 2, 6, 4, key="js_m_slider")
        
        st.session_state.n = n
        st.session_state.m = m
        
        np.random.seed(42)
        
        # Génération temps de traitement (avec quelques zéros)
        p_data = np.random.randint(0, 20, size=(m, n))
        # Assurer au moins 2 opérations par travail
        for j in range(n):
            non_zero = np.where(p_data[:, j] > 0)[0]
            if len(non_zero) < 2:
                indices = np.random.choice(m, 2, replace=False)
                for idx in indices:
                    p_data[idx, j] = np.random.randint(3, 15)
        
        rows = [f"M{i+1}" for i in range(m)]
        cols = [f"J{j+1}" for j in range(n)]
        st.session_state.edited_df = pd.DataFrame(p_data, index=rows, columns=cols)
        
        # Génération séquences de machines
        seq_data = np.zeros((m, n))
        for j in range(n):
            non_zero_machines = np.where(p_data[:, j] > 0)[0]
            num_ops = len(non_zero_machines)
            random_order = np.random.permutation(non_zero_machines) + 1  # 1-indexé
            seq_data[:num_ops, j] = random_order
        
        rows_seq = [f"Op{i+1}" for i in range(m)]
        st.session_state.machine_sequence_df = pd.DataFrame(seq_data.astype(int), index=rows_seq, columns=cols)
        
        col_display1, col_display2 = st.columns(2)
        with col_display1:
            st.markdown("#### Temps de Traitement")
            st.dataframe(st.session_state.edited_df, height=200)
        with col_display2:
            st.markdown("#### Séquence de Machines")
            st.dataframe(st.session_state.machine_sequence_df, height=200)
        
        # Aperçu des Itinéraires Interprétés
        st.markdown("---")
        st.markdown("#### 📋 Job Routes Interprétés:")
        proc_df = st.session_state.edited_df
        seq_df = st.session_state.machine_sequence_df
        max_ops = len(seq_df.index)
        
        for j in range(n):
            operations = []
            for op_idx in range(max_ops):
                machine_num = seq_df.iloc[op_idx, j]
                if float(machine_num) > 0:
                    machine_idx = int(machine_num) - 1
                    proc_time = proc_df.iloc[machine_idx, j]
                    if float(proc_time) > 0:
                        operations.append((machine_idx, float(proc_time)))
            
            if operations:
                route_str = " → ".join([f"M{m_idx+1}({t:.1f})" for m_idx, t in operations])
                st.success(f"**Job {j+1}:** {route_str}")
            else:
                st.warning(f"⚠️ **Job {j+1}:** Aucune opération valide.")
    
    elif input_mode == "Import Excel":
        st.subheader("📄 Import de Données Excel - Job Shop")
        st.info("""
        **Format Excel Requis:**
        
        **Feuille 1: "Temps_Traitement"** (Machines × Jobs, 0 signifie non utilisé)
        ```
               J1 J2 J3 J4 J5
        M1 4 6 8 6 4
        M2 5 7 3 5 6
        M3 3 2 5 3 8
        ```
        
        **Feuille 2: "Séquence_Machines"** (Opérations × Jobs, 1-indexé)
        ```
               J1 J2 J3 J4 J5
        Op1 2 1 3 2 1
        Op2 3 3 1 3 2
        Op3 1 2 2 1 3
        ```
        
        - **0 signifie pas d'opération** à cette position
        - Nombres **1, 2, 3** correspondent à **M1, M2, M3**
        - Opérations traitées dans l'ordre: Op1 → Op2 → Op3
        """)
        
        uploaded_file = st.file_uploader("Télécharger Fichier Excel", type=['xlsx'], key="js_upload")
        
        if uploaded_file:
            try:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # Vérifier feuilles requises
                proc_sheet = None
                seq_sheet = None
                
                # Chercher feuilles avec noms spécifiques ou motifs
                for sheet in sheet_names:
                    sheet_lower = sheet.lower()
                    if "traitement" in sheet_lower or "proc" in sheet_lower or "temps" in sheet_lower:
                        proc_sheet = sheet
                    if "sequence" in sheet_lower or "seq" in sheet_lower or "machine" in sheet_lower:
                        seq_sheet = sheet
                
                if not proc_sheet:
                    # Essayer de deviner - première feuille pour temps de traitement
                    proc_sheet = sheet_names[0]
                    st.warning(f"⚠️ Suppose que la feuille '{proc_sheet}' contient les temps de traitement")
                
                if not seq_sheet and len(sheet_names) > 1:
                    seq_sheet = sheet_names[1]
                    st.warning(f"⚠️ Suppose que la feuille '{seq_sheet}' contient la séquence de machines")
                
                if not seq_sheet:
                    st.error("❌ Impossible de trouver la feuille de séquence de machines!")
                    st.stop()
                
                # Lire données
                df_proc = pd.read_excel(uploaded_file, sheet_name=proc_sheet, index_col=0, header=0)
                df_seq = pd.read_excel(uploaded_file, sheet_name=seq_sheet, index_col=0, header=0)
                
                # Corriger noms de colonnes si numériques (1, 2, 3, ...)
                if df_proc.columns.dtype in ['int64', 'float64']:
                    df_proc.columns = [f"J{int(col)}" for col in df_proc.columns]
                
                if df_seq.columns.dtype in ['int64', 'float64']:
                    df_seq.columns = [f"J{int(col)}" for col in df_seq.columns]
                
                # Corriger noms d'index si numériques (pour df_proc, les index sont les machines)
                if df_proc.index.dtype in ['int64', 'float64']:
                    df_proc.index = [f"M{int(idx)}" for idx in df_proc.index]
                else:
                    df_proc.index = [str(idx).strip() for idx in df_proc.index]
                
                # Pour df_seq, les index sont les opérations (Op1, Op2, ...)
                if df_seq.index.dtype in ['int64', 'float64']:
                    df_seq.index = [f"Op{int(idx)}" for idx in df_seq.index]
                else:
                    df_seq.index = [str(idx).strip() for idx in df_seq.index]
                
                st.session_state.m = len(df_proc.index)
                st.session_state.n = len(df_proc.columns)
                st.session_state.edited_df = df_proc
                st.session_state.machine_sequence_df = df_seq
                
                st.success(f"✅ Chargé: **{st.session_state.n} Jobs** × **{st.session_state.m} Machines**")
                
                # Afficher les données avec formatage clair
                col_prev1, col_prev2 = st.columns(2)
                with col_prev1:
                    st.markdown("**Temps de Traitement (0 = opération non nécessaire):**")
                    st.dataframe(df_proc, height=200)
                    
                    # Explication
                    with st.expander("💡 Comprendre les Temps de Traitement"):
                        st.write("Chaque cellule montre le temps de traitement d'un job sur une machine.")
                        st.write("Exemple: M1-J1 = 4 signifie Job 1 prend 4 unités de temps sur Machine 1")
                        st.write("0 signifie que le job n'utilise pas cette machine")
                
                with col_prev2:
                    st.markdown("**Séquence de Machines (1-indexé, 0 = pas d'opération):**")
                    st.dataframe(df_seq, height=200)
                    
                    # Explication
                    with st.expander("💡 Comprendre la Séquence de Machines"):
                        st.write("Chaque colonne montre l'itinéraire pour un job.")
                        st.write("Exemple pour J1: Op1=2 → Op2=3 → Op3=1 signifie:")
                        st.write("1. Première opération: Machine 2")
                        st.write("2. Deuxième opération: Machine 3")
                        st.write("3. Troisième opération: Machine 1")
                        st.write("0 signifie pas d'opération à cette position")
                
                # Aperçu de l'interprétation des données
                st.markdown("---")
                st.markdown("#### 📝 Comment vos données seront traitées:")
                
                for j in range(st.session_state.n):
                    operations = []
                    for op_idx in range(len(df_seq.index)):
                        machine_num = df_seq.iloc[op_idx, j]
                        if pd.notna(machine_num) and float(machine_num) > 0:
                            machine_idx = int(machine_num) - 1
                            proc_time = df_proc.iloc[machine_idx, j]
                            if pd.notna(proc_time) and float(proc_time) > 0:
                                operations.append((machine_idx, float(proc_time)))
                    
                    if operations:
                        route_str = " → ".join([f"M{m+1}({t:.1f})" for m, t in operations])
                        st.success(f"**Job Route {j+1}:** {route_str}")
                    else:
                        st.warning(f"⚠️ **Job {j+1}:** Aucune opération valide!")
            
            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
                import traceback
                st.code(traceback.format_exc(), language='python')
                st.stop()
        else:
            st.info("Veuillez télécharger un fichier pour continuer.")
            st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# 12. LOGIQUE DE CALCUL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col_action, col_blank = st.columns([1, 4])
with col_action:
    calculate = st.button("🚀 Optimiser le Planning", type="primary", use_container_width=True)

if calculate:
    m = st.session_state.m
    n = st.session_state.n
    
    if n < 2 or m < 2:
        st.error("Vous avez besoin d'au moins 2 Jobs et 2 Machines pour la planification.")
        st.stop()
    
    if "Flow Shop" in shop_type:
        # ═══════════════════════════════════════════════════════════════════════
        # CALCUL FLOW SHOP
        # ═══════════════════════════════════════════════════════════════════════
        
        edited_df = st.session_state.edited_df
        with_setup = "Avec Préparation" in system_type
        
        if with_setup:
            if 'setup_matrices' not in st.session_state:
                st.error("⚠️ Veuillez configurer les matrices de préparation!")
                st.stop()
            setup_matrices = st.session_state.setup_matrices
        else:
            setup_matrices = None
        
        try:
            data_matrix = edited_df.values.astype(float)
            p_mat = data_matrix[:m, :].T
            d_arr = data_matrix[m, :]
            r_arr = np.zeros(n)
            
            results = []
            
            if with_setup:
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('SPT', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "SPT", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('LPT', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "LPT", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('EDD', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "EDD", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('WDD', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "WDD", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('TPS_asc', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "TPS Ascendant", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                seq, ms, C, S, setup_t = heuristic_schedule_with_setup('TPS_desc', p_mat, setup_matrices, m, n, r_arr, d_arr)
                results.append({"Method": "TPS Descendant", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                if m == 2:
                    seq, ms, C, S, setup_t = johnson_rule_with_setup(p_mat, setup_matrices, m, n, r_arr)
                    results.append({"Method": "Règle de Johnson", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                
                if m >= 3:
                    seq, ms, C, S, setup_t = cds_heuristic_with_setup(p_mat, setup_matrices, m, n, r_arr)
                    results.append({"Method": "CDS", "Seq": seq, "Makespan": ms, "C": C, "S": S, "Setup": setup_t})
                    
            else:
                seq, ms, C = heuristic_schedule('SPT', p_mat, m, n, r_arr, d_arr, None)
                results.append({"Method": "SPT", "Seq": seq, "Makespan": ms, "C": C})
                
                seq, ms, C = heuristic_schedule('LPT', p_mat, m, n, r_arr, d_arr, None)
                results.append({"Method": "LPT", "Seq": seq, "Makespan": ms, "C": C})
                
                seq, ms, C = heuristic_schedule('EDD', p_mat, m, n, r_arr, d_arr, None)
                results.append({"Method": "EDD", "Seq": seq, "Makespan": ms, "C": C})
                
                seq, ms, C = heuristic_schedule('WDD', p_mat, m, n, r_arr, d_arr, None)
                results.append({"Method": "WDD", "Seq": seq, "Makespan": ms, "C": C})
                
                if m == 2:
                    seq, ms, C = johnson_rule(p_mat, m, n, r_arr, None)
                    results.append({"Method": "Règle de Johnson", "Seq": seq, "Makespan": ms, "C": C})
                    
                if m >= 3:
                    num_k = m - 1
                    for k in range(1, num_k + 1):
                        A = np.sum(p_mat[:, :k], axis=1)
                        B = np.sum(p_mat[:, m-k:m], axis=1)
                        set1 = np.where(A < B)[0]
                        set2 = np.where(B <= A)[0]
                        seq = np.concatenate([set1[np.argsort(A[set1])], set2[np.argsort(-B[set2])]]).tolist()
                        ms, C_temp = simulate_flowshop(p_mat, seq, m, n, r_arr, None)
                        results.append({"Method": f"CDS (k={k})", "Seq": seq, "Makespan": ms, "C": C_temp})
            
            best_res = min(results, key=lambda x: x['Makespan'])
            
            st.markdown("### 🏆 Résultats d'Optimisation - Flow Shop")
            
            tft, tt, tar, tfr = compute_metrics(best_res['C'], r_arr, d_arr, n, m)
            
            if with_setup:
                total_setup = np.sum(best_res['Setup'])
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: 
                    render_metric_card("Meilleur Makespan", f"{best_res['Makespan']:.1f}", "#10b981")
                with m2: 
                    render_metric_card("Total Préparation", f"{total_setup:.1f}", "#f59e0b")
                with m3: 
                    render_metric_card("Retard Total", f"{tt:.1f}", "#ef4444")
                with m4: 
                    render_metric_card("Temps de Flux Moyen", f"{tfr:.1f}", "#3b82f6")
                with m5: 
                    render_metric_card("Meilleure Méthode", best_res['Method'], "#8b5cf6")
            else:
                m1, m2, m3, m4 = st.columns(4)
                with m1: 
                    render_metric_card("Meilleur Makespan", f"{best_res['Makespan']:.1f}", "#10b981")
                with m2: 
                    render_metric_card("Retard Total", f"{tt:.1f}", "#ef4444")
                with m3: 
                    render_metric_card("Temps de Flux Moyen", f"{tfr:.1f}", "#3b82f6")
                with m4: 
                    render_metric_card("Meilleure Méthode", best_res['Method'], "#8b5cf6")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📊 Diagrammes de Gantt", "📈 Comparaison", "💾 Export"])
            
            with tab1:
                st.markdown(f"#### Planning Gantt: {best_res['Method']}")
                
                if with_setup:
                    fig = plot_gantt_with_setup(best_res['Seq'], best_res['C'], best_res['S'],
                                               best_res['Setup'], p_mat, best_res['Makespan'], m, n, best_res['Method'])
                else:
                    fig = plot_gantt_matplotlib(best_res['Seq'], best_res['C'], p_mat, best_res['Makespan'], m, n, best_res['Method'])
                
                st.pyplot(fig)
                
                seq_1based = [i + 1 for i in best_res['Seq']]
                st.markdown(f"**📋 Séquence Optimale:** J{seq_1based} | **⏱️ Makespan:** {best_res['Makespan']:.2f}")
                
                st.markdown("---")
                
                with st.expander("📊 Comparer avec d'autres méthodes"):
                    for res in results:
                        if res != best_res:
                            if with_setup:
                                sub_fig = plot_gantt_with_setup(res['Seq'], res['C'], res['S'],
                                                               res['Setup'], p_mat, res['Makespan'], m, n, res['Method'])
                            else:
                                sub_fig = plot_gantt_matplotlib(res['Seq'], res['C'], p_mat, res['Makespan'], m, n, res['Method'])
                            st.pyplot(sub_fig)
                            st.markdown("---")
            
            with tab2:
                comp_data = []
                for res in results:
                    _tft, _tt, _tar, _tfr = compute_metrics(res['C'], r_arr, d_arr, n, m)
                    seq_str = " → ".join([f"J{i+1}" for i in res['Seq']])
                    comp_data.append({
                        "Algorithme": res['Method'],
                        "Makespan": f"{res['Makespan']:.2f}",
                        "Retard Total": f"{_tt:.2f}",
                        "Temps de Flux Moyen": f"{_tfr:.2f}",
                        "Séquence": seq_str
                    })
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, height=400)
            
            with tab3:
                csv = pd.DataFrame(comp_data).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Résultats CSV",
                    data=csv,
                    file_name='flowshop_results.csv',
                    mime='text/csv',
                )
        
        except Exception as e:
            st.error(f"Erreur de Calcul: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # ═══════════════════════════════════════════════════════════════════════
        # CALCUL JOB SHOP
        # ═══════════════════════════════════════════════════════════════════════
        
        try:
            # Parser les données d'entrée pour créer des objets JobShopJob
            proc_df = st.session_state.edited_df
            seq_df = st.session_state.machine_sequence_df
            
            # Convertir en tableaux numpy, gérant NaN
            proc_times = proc_df.fillna(0).values.astype(float)  # m × n
            machine_seq = seq_df.fillna(0).values.astype(float)  # max_ops × n
            
            jobs = []
            job_details = []
            
            for j in range(n):
                operations = []
                # Lire séquence de machines pour ce job
                for op_idx in range(machine_seq.shape[0]):
                    machine_num = machine_seq[op_idx, j]
                    if machine_num > 0:  # Ignorer 0/NaN
                        machine_idx = int(machine_num) - 1
                        if machine_idx < m:  # Vérifier index machine valide
                            proc_time = proc_times[machine_idx, j]
                            if proc_time > 0:
                                operations.append((machine_idx, proc_time))
                
                if operations:
                    jobs.append(JobShopJob(j, operations))
                    # Stocker détails pour affichage
                    route_str = " → ".join([f"M{m+1}({t})" for m, t in operations])
                    job_details.append(f"Job {j+1}: {route_str}")
                else:
                    st.warning(f"⚠️ Job {j+1} n'a aucune opération valide! Vérifiez données d'entrée.")
            
            if not jobs:
                st.error("❌ Aucun job valide trouvé! Vérifiez vos données d'entrée.")
                st.stop()
            
            # Afficher itinéraires de jobs
            st.markdown("#### 📋 Job Routes:")
            for detail in job_details:
                st.write(detail)
            
            # Exécuter algorithmes
            results = []
            
            # SPT
            seq, ms, schedule, gantt = jobshop_heuristic(jobs, 'SPT')
            results.append({
                "Method": "SPT",
                "Seq": seq,
                "Makespan": ms,
                "Schedule": schedule,
                "Gantt": gantt
            })
            
            # LPT
            seq, ms, schedule, gantt = jobshop_heuristic(jobs, 'LPT')
            results.append({
                "Method": "LPT",
                "Seq": seq,
                "Makespan": ms,
                "Schedule": schedule,
                "Gantt": gantt
            })
            
            # Jackson pour 2 machines seulement
            if m == 2:
                try:
                    seq_M1, seq_M2, _, _, _, _ = jackson_algorithm(jobs, m)
                    ms, schedule, gantt = simulate_jobshop_jackson(seq_M1, seq_M2, jobs)
                    # Seq approximative: ordre des jobs sur M1
                    seq = list(dict.fromkeys([jid for jid, _, _ in seq_M1]))
                    results.append({
                        "Method": "Jackson",
                        "Seq": seq,
                        "Makespan": ms,
                        "Schedule": schedule,
                        "Gantt": gantt
                    })
                except ValueError as ve:
                    st.warning(f"⚠️ Jackson non applicable: {ve}")
            
            # Trouver meilleur résultat
            best_res = min(results, key=lambda x: x['Makespan'])
            
            st.markdown("### 🏆 Résultats d'Optimisation - Job Shop")
            
            # Métriques principales
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Meilleur Makespan", f"{best_res['Makespan']:.1f}", "#10b981")
            with m2:
                total_ops = sum(len(job.operations) for job in jobs)
                render_metric_card("Opérations Totales", f"{total_ops}", "#3b82f6")
            with m3:
                render_metric_card("Meilleure Méthode", best_res['Method'], "#8b5cf6")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Onglets
            tab1, tab2, tab3 = st.tabs(["📊 Diagrammes de Gantt", "📈 Comparaison", "💾 Export Données"])
            
            with tab1:
                st.markdown(f"#### Planning Gantt: {best_res['Method']}")
                
                fig = plot_jobshop_gantt(
                    best_res['Gantt'],
                    jobs,
                    best_res['Makespan'],
                    m,
                    best_res['Method']
                )
                st.pyplot(fig)
                
                # Résumé
                seq_1based = [i + 1 for i in best_res['Seq']]
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"**📋 Séquence de Priorité:** J{seq_1based}")
                with col_s2:
                    st.markdown(f"**⏱️ Makespan:** {best_res['Makespan']:.2f}")
                
                st.markdown("---")
                
                # Afficher autres méthodes
                with st.expander("📊 Comparer avec d'autres méthodes"):
                    for res in results:
                        if res != best_res:
                            sub_fig = plot_jobshop_gantt(
                                res['Gantt'],
                                jobs,
                                res['Makespan'],
                                m,
                                res['Method']
                            )
                            st.pyplot(sub_fig)
                            
                            seq_comp = [i + 1 for i in res['Seq']]
                            st.markdown(f"**Priorité:** J{seq_comp} | **Makespan:** {res['Makespan']:.2f}")
                            st.markdown("---")
            
            with tab2:
                st.markdown("#### Planning Détailé - Meilleure Méthode")
            
            with tab3:
                st.markdown("#### Télécharger Résultats")
                
                # Tableau de comparaison
                comp_data = []
                for res in results:
                    seq_str = " → ".join([f"J{i+1}" for i in res['Seq']])
                    comp_data.append({
                        "Algorithme": res['Method'],
                        "Makespan": f"{res['Makespan']:.2f}",
                        "Séquence de Priorité": seq_str
                    })
                
                comp_df = pd.DataFrame(comp_data)
                st.markdown("**Tableau de Comparaison:**")
                st.dataframe(comp_df, use_container_width=True)
                
                csv_comp = comp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Comparaison CSV",
                    data=csv_comp,
                    file_name='jobshop_comparison.csv',
                    mime='text/csv',
                )
        
        except Exception as e:
            st.error(f"Erreur de Calcul Job Shop: {str(e)}")
            st.warning("Veuillez vérifier le format de vos données d'entrée.")
            import traceback
            st.code(traceback.format_exc())

