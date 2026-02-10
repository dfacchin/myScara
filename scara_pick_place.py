"""
SCARA Robot Dynamics Simulator - Full Featured with Variable Payload
=====================================================================
Sistema: Braccio SCARA orizzontale 2-DOF
Motori: MyActuator RMD-X8-25

FUNZIONALITÀ:
- Configurazione fissa (LEFT/RIGHT)
- Vincoli velocità e accelerazione
- Profilo trapezoidale o S-curve
- Pause configurabili ai waypoint
- Payload variabile (pick & place)
- Calcolo automatico tempi

Autore: Claude
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# =============================================================================
# CONFIGURAZIONI
# =============================================================================

class ArmConfig(Enum):
    LEFT = "left"    # q2 > 0
    RIGHT = "right"  # q2 < 0


class VelocityProfile(Enum):
    TRAPEZOIDAL = "trapezoidal"
    SCURVE = "scurve"


# =============================================================================
# PARAMETRI
# =============================================================================

@dataclass
class SCARAParams:
    """Parametri fisici del robot (senza payload, che è variabile)"""
    L1: float = 0.50
    L2: float = 0.50
    m1: float = 0.8
    m2: float = 0.4
    m_motor: float = 0.78
    Lc1: float = 0.25
    Lc2: float = 0.25
    
    gear_ratio: float = 9.0
    tau_peak: float = 25.0
    tau_nominal: float = 13.0
    J_rotor: float = 0.0002
    
    # Payload di default (può essere sovrascritto dai waypoint)
    default_payload: float = 0.0  # kg
    
    b1: float = 0.5
    b2: float = 0.3
    
    q2_min_margin: float = 15.0
    
    @property
    def I1(self):
        return (1/12) * self.m1 * self.L1**2
    
    @property
    def I2(self):
        return (1/12) * self.m2 * self.L2**2
    
    @property
    def J_motor_reflected(self):
        return self.J_rotor * self.gear_ratio**2
    
    @property
    def reach_max(self):
        return self.L1 + self.L2


@dataclass
class Waypoint:
    """
    Singolo waypoint con pausa e payload opzionali.
    
    Il payload specificato diventa attivo DOPO la pausa al waypoint.
    Questo simula:
    - Pick: arrivi al punto, pausa, PRENDI il pezzo (payload aumenta), riparti
    - Place: arrivi al punto, pausa, RILASCI il pezzo (payload diminuisce), riparti
    """
    x: float                    # Posizione X [cm]
    y: float                    # Posizione Y [cm]
    dwell: float = 0.0          # Tempo di pausa [s]
    payload: Optional[float] = None  # Payload DOPO questo waypoint [g] (None = mantieni precedente)
    
    def __repr__(self):
        parts = [f"({self.x}, {self.y}) cm"]
        if self.dwell > 0:
            parts.append(f"pausa {self.dwell}s")
        if self.payload is not None:
            parts.append(f"payload→{self.payload}g")
        return ", ".join(parts)


@dataclass 
class TrajectoryConstraints:
    """Vincoli sulla traiettoria"""
    max_velocity: float = 50.0        # cm/s
    max_acceleration: float = 100.0   # cm/s²
    max_jerk: float = 500.0           # cm/s³ (S-curve)
    profile_type: VelocityProfile = VelocityProfile.TRAPEZOIDAL
    dt: float = 0.001


# =============================================================================
# CINEMATICA
# =============================================================================

def forward_kinematics(q, params: SCARAParams):
    x = params.L1 * np.cos(q[0]) + params.L2 * np.cos(q[0] + q[1])
    y = params.L1 * np.sin(q[0]) + params.L2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def inverse_kinematics(x, y, params: SCARAParams, config: ArmConfig):
    L1, L2 = params.L1, params.L2
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)
    
    info = {'reachable': True, 'config_valid': True, 'q2_margin_deg': 0, 'message': ''}
    
    margin = 0.005
    if r > L1 + L2 - margin:
        info['reachable'] = False
        info['message'] = f"Fuori workspace (r={r*100:.1f} cm)"
        return None, info
    if r < abs(L1 - L2) + margin:
        info['reachable'] = False
        info['message'] = f"Troppo vicino (r={r*100:.1f} cm)"
        return None, info
    
    cos_q2 = np.clip((r_sq - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    sin_q2_abs = np.sqrt(1 - cos_q2**2)
    
    q2 = np.arctan2(sin_q2_abs if config == ArmConfig.LEFT else -sin_q2_abs, cos_q2)
    beta = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = np.arctan2(y, x) - beta
    
    q2_deg = np.rad2deg(q2)
    q2_margin = min(abs(q2_deg), 180 - abs(q2_deg))
    info['q2_margin_deg'] = q2_margin
    
    if q2_margin < params.q2_min_margin:
        info['config_valid'] = False
        info['message'] = f"Vicino singolarità (q2={q2_deg:.1f}°)"
    
    return np.array([q1, q2]), info


def compute_jacobian(q, params: SCARAParams):
    L1, L2 = params.L1, params.L2
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1*s1 - L2*s12, -L2*s12],
                     [L1*c1 + L2*c12,   L2*c12]])


def compute_jacobian_derivative(q, qd, params: SCARAParams):
    L1, L2 = params.L1, params.L2
    q1, q2 = q
    q1d, q2d = qd
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    return np.array([
        [-L1*c1*q1d - L2*c12*(q1d + q2d), -L2*c12*(q1d + q2d)],
        [-L1*s1*q1d - L2*s12*(q1d + q2d), -L2*s12*(q1d + q2d)]
    ])


def compute_manipulability(q, params: SCARAParams):
    J = compute_jacobian(q, params)
    return abs(np.linalg.det(J)) / (params.L1 * params.L2)


# =============================================================================
# DINAMICA CON PAYLOAD VARIABILE
# =============================================================================

def compute_inertia_matrix(q, params: SCARAParams, m_payload: float):
    """
    Matrice di inerzia M(q) con payload specificato.
    
    Args:
        q: angoli giunti [rad]
        params: parametri robot
        m_payload: massa payload [kg]
    """
    c2 = np.cos(q[1])
    p = params
    J_m = p.J_motor_reflected
    
    M11 = (p.I1 + p.I2 + 2*J_m + p.m1 * p.Lc1**2 
           + p.m2 * (p.L1**2 + p.Lc2**2 + 2*p.L1*p.Lc2*c2)
           + m_payload * (p.L1**2 + p.L2**2 + 2*p.L1*p.L2*c2)
           + p.m_motor * p.L1**2)
    M12 = (p.I2 + J_m + p.m2 * (p.Lc2**2 + p.L1*p.Lc2*c2)
           + m_payload * (p.L2**2 + p.L1*p.L2*c2))
    M22 = p.I2 + J_m + p.m2 * p.Lc2**2 + m_payload * p.L2**2
    
    return np.array([[M11, M12], [M12, M22]])


def compute_coriolis_matrix(q, qd, params: SCARAParams, m_payload: float):
    """Matrice di Coriolis C(q, q̇) con payload specificato."""
    s2 = np.sin(q[1])
    p = params
    h = p.m2 * p.L1 * p.Lc2 * s2 + m_payload * p.L1 * p.L2 * s2
    return np.array([[-h * qd[1], -h * (qd[0] + qd[1])],
                     [h * qd[0],   0]])


def inverse_dynamics(q, qd, qdd, params: SCARAParams, m_payload: float):
    """Dinamica inversa con payload specificato."""
    M = compute_inertia_matrix(q, params, m_payload)
    C = compute_coriolis_matrix(q, qd, params, m_payload)
    f = np.array([params.b1 * qd[0], params.b2 * qd[1]])
    return M @ qdd + C @ qd + f


# =============================================================================
# PROFILI DI MOVIMENTO
# =============================================================================

def generate_segment_profile(distance, v_max, a_max, dt):
    """Genera profilo trapezoidale per un singolo segmento."""
    if distance < 1e-6:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    
    v_max = abs(v_max)
    a_max = abs(a_max)
    
    t_acc = v_max / a_max
    s_acc = 0.5 * a_max * t_acc**2
    
    if 2 * s_acc > distance:
        t_acc = np.sqrt(distance / a_max)
        v_peak = a_max * t_acc
        t_cruise = 0
        t_total = 2 * t_acc
    else:
        v_peak = v_max
        s_cruise = distance - 2 * s_acc
        t_cruise = s_cruise / v_max
        t_total = 2 * t_acc + t_cruise
    
    t = np.arange(0, t_total + dt, dt)
    n = len(t)
    s = np.zeros(n)
    sd = np.zeros(n)
    sdd = np.zeros(n)
    
    for i, ti in enumerate(t):
        if ti <= t_acc:
            sdd[i] = a_max
            sd[i] = a_max * ti
            s[i] = 0.5 * a_max * ti**2
        elif ti <= t_acc + t_cruise:
            sdd[i] = 0
            sd[i] = v_peak
            s[i] = s_acc + v_peak * (ti - t_acc)
        else:
            t_dec = ti - t_acc - t_cruise
            sdd[i] = -a_max
            sd[i] = max(0, v_peak - a_max * t_dec)
            s[i] = min(distance, s_acc + v_peak * t_cruise + v_peak * t_dec - 0.5 * a_max * t_dec**2)
    
    return t, s, sd, sdd


def generate_dwell(duration, dt):
    """Genera periodo di pausa."""
    if duration < dt:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    
    t = np.arange(0, duration + dt, dt)
    n = len(t)
    return t, np.zeros(n), np.zeros(n), np.zeros(n)


# =============================================================================
# GENERAZIONE TRAIETTORIA COMPLETA
# =============================================================================

def generate_trajectory_with_payload(waypoints: List[Waypoint], params: SCARAParams,
                                      config: ArmConfig, constraints: TrajectoryConstraints):
    """
    Genera traiettoria completa con pause e payload variabile.
    
    Il payload cambia DOPO la pausa al waypoint dove è specificato.
    """
    dt = constraints.dt
    v_max = constraints.max_velocity / 100.0
    a_max = constraints.max_acceleration / 100.0
    
    n_wp = len(waypoints)
    
    # 1. Valida waypoint e costruisci profilo payload
    print("\n🎯 VALIDAZIONE WAYPOINT:")
    config_name = "SINISTRO" if config == ArmConfig.LEFT else "DESTRO"
    print(f"   Configurazione: {config_name}")
    print(f"   Vincoli: V_max={constraints.max_velocity} cm/s, A_max={constraints.max_acceleration} cm/s²")
    print()
    
    wp_coords = []
    payload_at_wp = []  # Payload attivo DOPO ogni waypoint
    current_payload = params.default_payload
    
    for i, wp in enumerate(waypoints):
        x_m, y_m = wp.x / 100.0, wp.y / 100.0
        q_test, info = inverse_kinematics(x_m, y_m, params, config)
        
        # Aggiorna payload se specificato
        if wp.payload is not None:
            current_payload = wp.payload / 1000.0  # g -> kg
        payload_at_wp.append(current_payload)
        
        # Costruisci stringa descrittiva
        desc_parts = []
        if wp.dwell > 0:
            desc_parts.append(f"pausa {wp.dwell}s")
        if wp.payload is not None:
            desc_parts.append(f"payload→{wp.payload}g")
        desc = ", " + ", ".join(desc_parts) if desc_parts else ""
        
        if q_test is None or not info['config_valid']:
            print(f"   ✗ WP{i}: ({wp.x}, {wp.y}) cm{desc} - {info['message']}")
            return None, None, None, None, None, False
        
        print(f"   ✓ WP{i}: ({wp.x}, {wp.y}) cm{desc} - q2={np.rad2deg(q_test[1]):.1f}°")
        wp_coords.append((x_m, y_m))
    
    # 2. Calcola geometria
    segment_lengths = []
    for i in range(n_wp - 1):
        dx = wp_coords[i+1][0] - wp_coords[i][0]
        dy = wp_coords[i+1][1] - wp_coords[i][1]
        segment_lengths.append(np.sqrt(dx**2 + dy**2))
    
    total_path_length = sum(segment_lengths)
    total_dwell_time = sum(wp.dwell for wp in waypoints)
    
    print(f"\n📏 GEOMETRIA:")
    print(f"   Lunghezza percorso: {total_path_length*100:.1f} cm")
    print(f"   Tempo pause totale: {total_dwell_time:.2f} s")
    
    print(f"\n📦 PROFILO PAYLOAD:")
    for i, (wp, pl) in enumerate(zip(waypoints, payload_at_wp)):
        change = ""
        if wp.payload is not None:
            if i == 0:
                change = " (iniziale)"
            else:
                prev_pl = payload_at_wp[i-1] if i > 0 else params.default_payload
                if pl > prev_pl:
                    change = f" ← PICK (+{(pl-prev_pl)*1000:.0f}g)"
                elif pl < prev_pl:
                    change = f" ← PLACE (-{(prev_pl-pl)*1000:.0f}g)"
        print(f"   Dopo WP{i}: {pl*1000:.0f}g{change}")
    
    # 3. Genera profili movimento
    print(f"\n⚙️  Generazione profili di movimento...")
    
    all_t = []
    all_s = []
    all_sd = []
    all_sdd = []
    all_x = []
    all_y = []
    all_payload = []  # Payload attivo in ogni istante
    
    current_time = 0
    current_s = 0
    active_payload = params.default_payload
    
    waypoint_times = []
    waypoint_arrival_times = []
    waypoint_departure_times = []
    
    for seg_idx in range(n_wp):
        wp = waypoints[seg_idx]
        
        # Registra arrivo
        waypoint_arrival_times.append(current_time)
        
        # Pausa al waypoint
        if wp.dwell > 0:
            t_dwell, s_dwell, sd_dwell, sdd_dwell = generate_dwell(wp.dwell, dt)
            
            x_pos, y_pos = wp_coords[seg_idx]
            
            # Prima metà della pausa: payload precedente
            # Seconda metà (o dopo): nuovo payload (se cambia)
            half_dwell_samples = len(t_dwell) // 2
            
            for i in range(len(t_dwell)):
                if i == 0 and len(all_t) > 0:
                    continue  # Salta primo campione se non è l'inizio
                
                all_t.append(current_time + t_dwell[i])
                all_s.append(current_s)
                all_sd.append(0)
                all_sdd.append(0)
                all_x.append(x_pos)
                all_y.append(y_pos)
                
                # Cambio payload a metà pausa
                if i >= half_dwell_samples and wp.payload is not None:
                    active_payload = wp.payload / 1000.0
                all_payload.append(active_payload)
            
            current_time += t_dwell[-1]
        else:
            # No pausa: aggiorna payload immediatamente se specificato
            if wp.payload is not None:
                active_payload = wp.payload / 1000.0
        
        waypoint_departure_times.append(current_time)
        waypoint_times.append(current_time)
        
        # Movimento al waypoint successivo (se non è l'ultimo)
        if seg_idx < n_wp - 1:
            dist = segment_lengths[seg_idx]
            t_seg, s_seg, sd_seg, sdd_seg = generate_segment_profile(dist, v_max, a_max, dt)
            
            x_start, y_start = wp_coords[seg_idx]
            x_end, y_end = wp_coords[seg_idx + 1]
            
            if dist > 1e-6:
                ratio = s_seg / dist
            else:
                ratio = np.ones_like(s_seg)
            
            x_seg = x_start + (x_end - x_start) * ratio
            y_seg = y_start + (y_end - y_start) * ratio
            
            start_idx = 1 if len(all_t) > 0 else 0
            
            for i in range(start_idx, len(t_seg)):
                all_t.append(current_time + t_seg[i])
                all_s.append(current_s + s_seg[i])
                all_sd.append(sd_seg[i])
                all_sdd.append(sdd_seg[i])
                all_x.append(x_seg[i])
                all_y.append(y_seg[i])
                all_payload.append(active_payload)
            
            current_time += t_seg[-1]
            current_s += dist
    
    # Converti in array
    t = np.array(all_t)
    s = np.array(all_s)
    sd = np.array(all_sd)
    sdd = np.array(all_sdd)
    x_ref = np.array(all_x)
    y_ref = np.array(all_y)
    payload_profile = np.array(all_payload)
    
    n = len(t)
    
    print(f"   Punti generati: {n}")
    print(f"   Tempo totale: {t[-1]:.2f} s")
    
    # 4. Velocità e accelerazione cartesiane
    # Usa derivate analitiche dal profilo di movimento, non gradient numerico
    # che crea spike alle discontinuità
    
    is_paused = sd < 1e-6
    
    # Direzione tangente in ogni punto
    dx = np.gradient(x_ref, edge_order=2)
    dy = np.gradient(y_ref, edge_order=2)
    ds = np.sqrt(dx**2 + dy**2)
    ds[ds < 1e-10] = 1e-10  # Evita divisione per zero
    
    # Versore tangente
    tx = dx / ds
    ty = dy / ds
    
    # Velocità = versore tangente * velocità scalare
    xd_ref = tx * sd
    yd_ref = ty * sd
    
    # Accelerazione tangenziale e centripeta
    # Per semplicità usiamo solo la componente tangenziale
    xdd_ref = tx * sdd
    ydd_ref = ty * sdd
    
    # Azzera durante le pause
    xd_ref[is_paused] = 0
    yd_ref[is_paused] = 0
    xdd_ref[is_paused] = 0
    ydd_ref[is_paused] = 0
    
    # 5. Cinematica inversa
    print(f"\n🔄 Conversione cinematica inversa...")
    
    q = np.zeros((2, n))
    qd = np.zeros((2, n))
    qdd = np.zeros((2, n))
    manipulability = np.zeros(n)
    q2_margin = np.zeros(n)
    
    for i in range(n):
        q_i, info = inverse_kinematics(x_ref[i], y_ref[i], params, config)
        
        if q_i is None:
            print(f"   ✗ Errore a t={t[i]:.3f}s")
            return None, None, None, None, None, False
        
        q[:, i] = q_i
        q2_margin[i] = info['q2_margin_deg']
        manipulability[i] = compute_manipulability(q_i, params)
        
        J = compute_jacobian(q_i, params)
        cart_vel = np.array([xd_ref[i], yd_ref[i]])
        
        # Se velocità cartesiana è quasi zero, velocità giunto è zero
        if np.linalg.norm(cart_vel) < 1e-8:
            qd[:, i] = np.zeros(2)
        else:
            try:
                qd[:, i] = np.linalg.solve(J, cart_vel)
            except:
                qd[:, i] = np.zeros(2)
        
        Jdot = compute_jacobian_derivative(q_i, qd[:, i], params)
        cart_acc = np.array([xdd_ref[i], ydd_ref[i]])
        
        # Se accelerazione cartesiana è quasi zero, accelerazione giunto è zero
        if np.linalg.norm(cart_acc) < 1e-8 and np.linalg.norm(Jdot @ qd[:, i]) < 1e-8:
            qdd[:, i] = np.zeros(2)
        else:
            try:
                qdd[:, i] = np.linalg.solve(J, cart_acc - Jdot @ qd[:, i])
            except:
                qdd[:, i] = np.zeros(2)
    
    # Verifica configurazione
    if config == ArmConfig.LEFT and np.any(q[1] < -0.01):
        print(f"   ✗ Cambio configurazione!")
        return None, None, None, None, None, False
    if config == ArmConfig.RIGHT and np.any(q[1] > 0.01):
        print(f"   ✗ Cambio configurazione!")
        return None, None, None, None, None, False
    
    # Azzera velocità e accelerazioni durante le pause PRIMA del filtro
    qd[0][is_paused] = 0
    qd[1][is_paused] = 0
    qdd[0][is_paused] = 0
    qdd[1][is_paused] = 0
    
    # Filtra per rimuovere spike numerici ai bordi movimento/pausa
    from scipy.ndimage import uniform_filter1d
    window = 7
    
    # Filtra velocità e accelerazioni
    qd_filt = np.zeros_like(qd)
    qdd_filt = np.zeros_like(qdd)
    for i in range(2):
        qd_filt[i] = uniform_filter1d(qd[i], window, mode='nearest')
        qdd_filt[i] = uniform_filter1d(qdd[i], window, mode='nearest')
    
    # Mantieni zero durante le pause
    qd_filt[0][is_paused] = 0
    qd_filt[1][is_paused] = 0
    qdd_filt[0][is_paused] = 0
    qdd_filt[1][is_paused] = 0
    
    qd = qd_filt
    qdd = qdd_filt
    
    print(f"   ✓ Completato")
    print(f"   Margine min singolarità: {np.min(q2_margin):.1f}°")
    
    cart = {
        'x': x_ref, 'y': y_ref,
        'xd': xd_ref, 'yd': yd_ref,
        'xdd': xdd_ref, 'ydd': ydd_ref,
        's': s, 'sd': sd, 'sdd': sdd,
        'manipulability': manipulability,
        'q2_margin': q2_margin,
        'payload': payload_profile,
        'waypoint_times': waypoint_times,
        'waypoint_arrival_times': waypoint_arrival_times,
        'waypoint_departure_times': waypoint_departure_times,
        'is_paused': is_paused
    }
    
    return t, q, qd, qdd, cart, True


# =============================================================================
# SIMULAZIONE CON PAYLOAD VARIABILE
# =============================================================================

def simulate_trajectory(t, q, qd, qdd, payload_profile, params: SCARAParams):
    """Simula dinamica con payload variabile nel tempo."""
    n = len(t)
    tau = np.zeros((2, n))
    M_trace = np.zeros((2, 2, n))
    
    for i in range(n):
        m_payload = payload_profile[i]
        tau[:, i] = inverse_dynamics(q[:, i], qd[:, i], qdd[:, i], params, m_payload)
        M_trace[:, :, i] = compute_inertia_matrix(q[:, i], params, m_payload)
    
    return tau, M_trace


def analyze_feasibility(tau, params: SCARAParams):
    tau1_max = np.max(np.abs(tau[0]))
    tau2_max = np.max(np.abs(tau[1]))
    tau1_rms = np.sqrt(np.mean(tau[0]**2))
    tau2_rms = np.sqrt(np.mean(tau[1]**2))
    
    return {
        'tau1_peak': tau1_max,
        'tau2_peak': tau2_max,
        'tau1_rms': tau1_rms,
        'tau2_rms': tau2_rms,
        'peak_ok': max(tau1_max, tau2_max) < params.tau_peak,
        'continuous_ok': max(tau1_rms, tau2_rms) < params.tau_nominal,
    }


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def plot_results(t, q, qd, qdd, tau, cart, M_trace, params, config, constraints,
                 feasibility, waypoints: List[Waypoint]):
    
    config_name = "SINISTRO" if config == ArmConfig.LEFT else "DESTRO"
    total_dwell = sum(wp.dwell for wp in waypoints)
    max_payload = np.max(cart['payload']) * 1000
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Traiettoria SCARA Pick & Place - Braccio {config_name}\n'
                 f'V_max={constraints.max_velocity} cm/s, Payload max={max_payload:.0f}g', 
                 fontsize=12, fontweight='bold')
    
    wp_x = [wp.x for wp in waypoints]
    wp_y = [wp.y for wp in waypoints]
    
    # 1. Traiettoria XY con indicazione payload
    ax1 = fig.add_subplot(4, 4, 1)
    theta = np.linspace(0, 2*np.pi, 100)
    r_max = params.reach_max * 100
    ax1.plot(r_max * np.cos(theta), r_max * np.sin(theta), 'g--', alpha=0.3)
    ax1.fill(r_max * np.cos(theta), r_max * np.sin(theta), alpha=0.05, color='green')
    
    # Colora traiettoria in base al payload
    for i in range(len(t)-1):
        color = plt.cm.Reds(cart['payload'][i] / max(max_payload/1000, 0.001) * 0.8 + 0.1)
        ax1.plot(cart['x'][i:i+2]*100, cart['y'][i:i+2]*100, color=color, linewidth=2.5)
    
    # Waypoint
    for i, wp in enumerate(waypoints):
        if wp.payload is not None and wp.payload > 0:
            marker, color, size = 's', 'red', 14
        elif wp.payload is not None and wp.payload == 0:
            marker, color, size = 'D', 'blue', 12
        else:
            marker, color, size = 'o', 'gray', 8
        ax1.plot(wp.x, wp.y, marker, color=color, markersize=size, zorder=5)
        
        label = f'{i}'
        if wp.payload is not None:
            label += f'\n{wp.payload}g'
        ax1.annotate(label, (wp.x, wp.y), xytext=(8, 8), 
                    textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax1.plot(0, 0, 'ks', markersize=12)
    ax1.set_xlabel('X [cm]')
    ax1.set_ylabel('Y [cm]')
    ax1.set_title('Traiettoria (■=pick, ◆=place)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Profilo payload nel tempo
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.fill_between(t, 0, cart['payload']*1000, alpha=0.6, color='coral', step='post')
    ax2.plot(t, cart['payload']*1000, 'r-', linewidth=2, drawstyle='steps-post')
    
    for i, wp in enumerate(waypoints):
        if wp.payload is not None:
            t_wp = cart['waypoint_departure_times'][i]
            ax2.axvline(x=t_wp, color='gray', linestyle=':', alpha=0.5)
            ax2.annotate(f'WP{i}', (t_wp, wp.payload*1.05), fontsize=8)
    
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel('Payload [g]')
    ax2.set_title('Profilo Payload')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # 3. Posizione X, Y
    ax3 = fig.add_subplot(4, 4, 3)
    for i, wp in enumerate(waypoints):
        if wp.dwell > 0:
            ax3.axvspan(cart['waypoint_arrival_times'][i], 
                       cart['waypoint_departure_times'][i], 
                       alpha=0.3, color='yellow')
    ax3.plot(t, cart['x']*100, 'b-', label='X', linewidth=1.5)
    ax3.plot(t, cart['y']*100, 'r-', label='Y', linewidth=1.5)
    ax3.set_xlabel('Tempo [s]')
    ax3.set_ylabel('Posizione [cm]')
    ax3.set_title('Coordinate Cartesiane')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Velocità
    ax4 = fig.add_subplot(4, 4, 4)
    for i, wp in enumerate(waypoints):
        if wp.dwell > 0:
            ax4.axvspan(cart['waypoint_arrival_times'][i], 
                       cart['waypoint_departure_times'][i], 
                       alpha=0.3, color='yellow')
    ax4.plot(t, cart['sd']*100, 'b-', linewidth=2)
    ax4.fill_between(t, 0, cart['sd']*100, alpha=0.3)
    ax4.axhline(y=constraints.max_velocity, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Tempo [s]')
    ax4.set_ylabel('Velocità [cm/s]')
    ax4.set_title('Profilo Velocità')
    ax4.grid(True, alpha=0.3)
    
    # 5. Angoli giunti
    ax5 = fig.add_subplot(4, 4, 5)
    for i, wp in enumerate(waypoints):
        if wp.dwell > 0:
            ax5.axvspan(cart['waypoint_arrival_times'][i], 
                       cart['waypoint_departure_times'][i], 
                       alpha=0.3, color='yellow')
    ax5.plot(t, np.rad2deg(q[0]), 'b-', label='q₁', linewidth=1.5)
    ax5.plot(t, np.rad2deg(q[1]), 'r-', label='q₂', linewidth=1.5)
    ax5.set_xlabel('Tempo [s]')
    ax5.set_ylabel('Angolo [°]')
    ax5.set_title('Posizioni Giunti')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Velocità giunti
    ax6 = fig.add_subplot(4, 4, 6)
    for i, wp in enumerate(waypoints):
        if wp.dwell > 0:
            ax6.axvspan(cart['waypoint_arrival_times'][i], 
                       cart['waypoint_departure_times'][i], 
                       alpha=0.3, color='yellow')
    ax6.plot(t, np.rad2deg(qd[0]), 'b-', label='q̇₁', linewidth=1.5)
    ax6.plot(t, np.rad2deg(qd[1]), 'r-', label='q̇₂', linewidth=1.5)
    ax6.set_xlabel('Tempo [s]')
    ax6.set_ylabel('Velocità [°/s]')
    ax6.set_title('Velocità Giunti')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Coppie con evidenza effetto payload
    ax7 = fig.add_subplot(4, 4, 7)
    
    # Colora sfondo in base al payload
    for i in range(len(t)-1):
        if cart['payload'][i] > 0.001:
            ax7.axvspan(t[i], t[i+1], alpha=0.1, color='red')
    
    for i, wp in enumerate(waypoints):
        if wp.dwell > 0:
            ax7.axvspan(cart['waypoint_arrival_times'][i], 
                       cart['waypoint_departure_times'][i], 
                       alpha=0.3, color='yellow')
    
    ax7.plot(t, tau[0], 'b-', label='τ₁', linewidth=1.5)
    ax7.plot(t, tau[1], 'r-', label='τ₂', linewidth=1.5)
    ax7.axhline(y=params.tau_peak, color='k', linestyle='--', alpha=0.5)
    ax7.axhline(y=-params.tau_peak, color='k', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Tempo [s]')
    ax7.set_ylabel('Coppia [N·m]')
    ax7.set_title('Coppie (sfondo rosso = con carico)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Inerzia M11 (mostra effetto payload)
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.plot(t, M_trace[0, 0], 'purple', linewidth=2, label='M₁₁')
    ax8.fill_between(t, 0, M_trace[0, 0], alpha=0.3, color='purple')
    ax8.set_xlabel('Tempo [s]')
    ax8.set_ylabel('Inerzia [kg·m²]')
    ax8.set_title('Inerzia M₁₁ (varia con payload)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Sequenza movimento
    ax9 = fig.add_subplot(4, 4, 9)
    theta = np.linspace(0, 2*np.pi, 100)
    ax9.plot(params.reach_max*100 * np.cos(theta), params.reach_max*100 * np.sin(theta), 
             'g--', alpha=0.3)
    
    n_frames = 10
    indices = np.linspace(0, len(t)-1, n_frames).astype(int)
    
    for i, idx in enumerate(indices):
        q1, q2 = q[0, idx], q[1, idx]
        x1 = params.L1 * np.cos(q1) * 100
        y1 = params.L1 * np.sin(q1) * 100
        x2 = x1 + params.L2 * np.cos(q1 + q2) * 100
        y2 = y1 + params.L2 * np.sin(q1 + q2) * 100
        
        # Colore in base al payload
        color = plt.cm.viridis(i / n_frames)
        lw = 5 if cart['payload'][idx] > 0.001 else 3
        
        ax9.plot([0, x1], [0, y1], color=color, linewidth=lw, alpha=0.8)
        ax9.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.8)
        
        # Indicatore payload
        if cart['payload'][idx] > 0.001:
            ax9.plot(x2, y2, 'ro', markersize=8)
    
    ax9.plot(0, 0, 'ks', markersize=12)
    ax9.set_xlabel('X [cm]')
    ax9.set_ylabel('Y [cm]')
    ax9.set_title('Sequenza (●=con carico)')
    ax9.axis('equal')
    ax9.grid(True, alpha=0.3)
    
    # 10. Utilizzo motori
    ax10 = fig.add_subplot(4, 4, 10)
    ratio1 = np.abs(tau[0]) / params.tau_peak * 100
    ratio2 = np.abs(tau[1]) / params.tau_peak * 100
    ax10.fill_between(t, 0, ratio1, alpha=0.5, label='Motor 1')
    ax10.fill_between(t, 0, ratio2, alpha=0.5, label='Motor 2')
    ax10.axhline(y=100, color='r', linestyle='--')
    ax10.set_xlabel('Tempo [s]')
    ax10.set_ylabel('Utilizzo [%]')
    ax10.set_title('Carico Motori')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim(0, max(20, np.max([ratio1, ratio2])*1.2))
    
    # 11. Margine singolarità
    ax11 = fig.add_subplot(4, 4, 11)
    ax11.fill_between(t, 0, cart['q2_margin'], alpha=0.6, color='teal')
    ax11.axhline(y=params.q2_min_margin, color='r', linestyle='--', linewidth=2)
    ax11.set_xlabel('Tempo [s]')
    ax11.set_ylabel('Margine [°]')
    ax11.set_title('Distanza Singolarità')
    ax11.grid(True, alpha=0.3)
    
    # 12. Confronto coppia con/senza carico
    ax12 = fig.add_subplot(4, 4, 12)
    
    # Trova campioni con e senza carico durante movimento
    moving = ~cart['is_paused']
    with_load = (cart['payload'] > 0.001) & moving
    without_load = (cart['payload'] <= 0.001) & moving
    
    if np.any(with_load) and np.any(without_load):
        tau1_with = np.abs(tau[0][with_load])
        tau1_without = np.abs(tau[0][without_load])
        
        ax12.boxplot([tau1_without, tau1_with], labels=['Senza carico', 'Con carico'])
        ax12.set_ylabel('|τ₁| [N·m]')
        ax12.set_title('Effetto Payload su Coppia')
        ax12.grid(True, alpha=0.3)
    else:
        ax12.text(0.5, 0.5, 'Dati insufficienti\nper confronto', 
                 ha='center', va='center', transform=ax12.transAxes)
        ax12.set_title('Confronto Coppia')
    
    # 13-16. Report
    ax13 = fig.add_subplot(4, 4, 13)
    ax13.axis('off')
    
    # Timeline dettagliata
    timeline_str = ""
    for i, wp in enumerate(waypoints):
        arr = cart['waypoint_arrival_times'][i]
        dep = cart['waypoint_departure_times'][i]
        
        line = f"WP{i} ({wp.x},{wp.y}): {arr:.2f}s"
        if wp.dwell > 0:
            line += f" → pausa {wp.dwell}s"
        if wp.payload is not None:
            line += f" → {wp.payload}g"
        line += f" → {dep:.2f}s\n"
        timeline_str += line
    
    text1 = f"""
══════════════════════════════════════
         TIMELINE DETTAGLIATA
══════════════════════════════════════

{timeline_str}
Tempo movimento: {t[-1] - total_dwell:.2f} s
Tempo pause:     {total_dwell:.2f} s
TEMPO TOTALE:    {t[-1]:.2f} s
"""
    
    ax13.text(0.02, 0.98, text1, transform=ax13.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax14 = fig.add_subplot(4, 4, 14)
    ax14.axis('off')
    
    text2 = f"""
══════════════════════════════════════
           ANALISI PAYLOAD
══════════════════════════════════════

Payload minimo:  {np.min(cart['payload'])*1000:.0f} g
Payload massimo: {np.max(cart['payload'])*1000:.0f} g

Inerzia M₁₁:
  Min: {np.min(M_trace[0,0]):.4f} kg·m²
  Max: {np.max(M_trace[0,0]):.4f} kg·m²
  Δ:   {(np.max(M_trace[0,0])-np.min(M_trace[0,0]))*100/np.min(M_trace[0,0]):.1f}%

══════════════════════════════════════
           ANALISI MOTORI
══════════════════════════════════════

τ₁ picco: {feasibility['tau1_peak']:.3f} N·m
τ₂ picco: {feasibility['tau2_peak']:.3f} N·m
τ₁ RMS:   {feasibility['tau1_rms']:.3f} N·m
τ₂ RMS:   {feasibility['tau2_rms']:.3f} N·m

Limite picco ({params.tau_peak} N·m): {'✓ OK' if feasibility['peak_ok'] else '✗ SUPERATO'}
Limite cont. ({params.tau_nominal} N·m): {'✓ OK' if feasibility['continuous_ok'] else '✗ SUPERATO'}
"""
    
    ax14.text(0.02, 0.98, text2, transform=ax14.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Grafico a barre payload per waypoint
    ax15 = fig.add_subplot(4, 4, 15)
    wp_indices = range(len(waypoints))
    
    # Costruisci array payload effettivo dopo ogni waypoint
    effective_payload = []
    current = params.default_payload * 1000
    for wp in waypoints:
        if wp.payload is not None:
            current = wp.payload
        effective_payload.append(current)
    
    colors = ['coral' if p > 0 else 'lightblue' for p in effective_payload]
    ax15.bar(wp_indices, effective_payload, color=colors, edgecolor='black')
    ax15.set_xlabel('Waypoint')
    ax15.set_ylabel('Payload [g]')
    ax15.set_title('Payload per Waypoint')
    ax15.set_xticks(wp_indices)
    ax15.grid(True, alpha=0.3, axis='y')
    
    # Grafico tempo per segmento
    ax16 = fig.add_subplot(4, 4, 16)
    segment_times = []
    for i in range(len(waypoints)-1):
        t_start = cart['waypoint_departure_times'][i]
        t_end = cart['waypoint_arrival_times'][i+1]
        segment_times.append(t_end - t_start)
    
    dwell_times = [wp.dwell for wp in waypoints]
    
    x_pos = np.arange(len(waypoints))
    width = 0.35
    
    ax16.bar(x_pos[:-1] + width/2, segment_times, width, label='Movimento', color='steelblue')
    ax16.bar(x_pos, dwell_times, width, label='Pausa', color='gold')
    ax16.set_xlabel('Waypoint')
    ax16.set_ylabel('Tempo [s]')
    ax16.set_title('Tempi per Segmento')
    ax16.set_xticks(x_pos)
    ax16.legend()
    ax16.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("  SIMULATORE SCARA - PICK & PLACE CON PAYLOAD VARIABILE")
    print("="*70)
    
    params = SCARAParams()
    
    # =========================================================================
    # CONFIGURAZIONE - MODIFICA QUI
    # =========================================================================
    
    ARM_CONFIG = ArmConfig.LEFT
    
    constraints = TrajectoryConstraints(
        max_velocity=60.0,           # cm/s
        max_acceleration=120.0,      # cm/s²
        profile_type=VelocityProfile.TRAPEZOIDAL,
        dt=0.001
    )
    
    # Waypoint con pause e payload
    # payload=None significa "mantieni il payload precedente"
    # payload=X significa "dopo questo waypoint il payload diventa X grammi"
    
    waypoints = [
        # Home - partenza senza carico
        Waypoint(x=75, y=15, dwell=0, payload=0),
        
        # Avvicinamento al pick point
        Waypoint(x=50, y=55, dwell=0.2, payload=None),   # Breve pausa per stabilizzare
        
        # PICK - prendi il pezzo (300g)
        Waypoint(x=35, y=60, dwell=0.5, payload=300),    # Pausa + pick
        
        # Solleva e trasporta
        Waypoint(x=50, y=50, dwell=0, payload=None),     # Transito
        
        # Avvicinamento al place point  
        Waypoint(x=70, y=40, dwell=0.2, payload=None),   # Stabilizza
        
        # PLACE - rilascia il pezzo
        Waypoint(x=75, y=35, dwell=0.5, payload=0),      # Pausa + rilascio
        
        # Ritorno home
        Waypoint(x=75, y=15, dwell=0, payload=None),
    ]
    
    # =========================================================================
    
    print(f"\n📐 ROBOT:")
    print(f"   Link: {params.L1*100:.0f} + {params.L2*100:.0f} cm")
    print(f"   Coppia max: {params.tau_peak} N·m")
    
    print(f"\n📍 WAYPOINT:")
    for i, wp in enumerate(waypoints):
        print(f"   {i}: {wp}")
    
    # Genera traiettoria
    t, q, qd, qdd, cart, success = generate_trajectory_with_payload(
        waypoints, params, ARM_CONFIG, constraints
    )
    
    if not success:
        print("\n❌ Traiettoria non valida!")
        exit(1)
    
    # Simula con payload variabile
    print(f"\n🔄 Simulazione dinamica con payload variabile...")
    tau, M_trace = simulate_trajectory(t, q, qd, qdd, cart['payload'], params)
    feasibility = analyze_feasibility(tau, params)
    
    # Risultati
    total_dwell = sum(wp.dwell for wp in waypoints)
    print(f"\n{'='*70}")
    print(f"  RISULTATI")
    print(f"{'='*70}")
    print(f"\n   Tempo movimento: {t[-1] - total_dwell:.2f} s")
    print(f"   Tempo pause: {total_dwell:.2f} s")
    print(f"   TEMPO CICLO: {t[-1]:.2f} s")
    
    print(f"\n   Payload: {np.min(cart['payload'])*1000:.0f}g - {np.max(cart['payload'])*1000:.0f}g")
    print(f"   Coppia max τ₁: {feasibility['tau1_peak']:.3f} N·m")
    print(f"   Coppia max τ₂: {feasibility['tau2_peak']:.3f} N·m")
    print(f"   Verifica motori: {'✓ OK' if feasibility['peak_ok'] else '✗ SUPERATO'}")
    
    # Plot
    print(f"\n📈 Generazione grafici...")
    fig = plot_results(t, q, qd, qdd, tau, cart, M_trace, params, ARM_CONFIG,
                       constraints, feasibility, waypoints)
    
    output_path = '/mnt/user-data/outputs/scara_pick_place.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   Salvato: {output_path}")
    
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"  ✅ COMPLETATO")
    print(f"{'='*70}\n")
