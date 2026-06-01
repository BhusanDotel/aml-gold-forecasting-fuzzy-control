"""
Task 2, Part 1 — Fuzzy Logic Controller (FLC) for an Intelligent Assistive Care Environment.

System: Smart flat for disabled residents.
Model: Mamdani Fuzzy Inference System.

Controller 1 — HVAC Temperature Control
  Inputs:
    temperature    (°C):   0 – 40
    humidity       (%):    0 – 100
    time_of_day    (h):    0 – 24
  Output:
    hvac_power     (-100 to +100):  negative = cooling, positive = heating

Controller 2 — Lighting Control
  Inputs:
    ambient_light  (lux):  0 – 1000
    time_of_day    (h):    0 – 24
  Output:
    light_level    (%):    0 – 100

Both use triangular / trapezoidal membership functions and centroid defuzzification.
"""

import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)


# ── Universe of discourse ────────────────────────────────────────────────────

temp_uni   = np.arange(0, 41, 0.5)
hum_uni    = np.arange(0, 101, 1.0)
tod_uni    = np.arange(0, 25, 0.25)
hvac_uni   = np.arange(-100, 101, 1.0)
light_uni  = np.arange(0, 1001, 5.0)
ll_uni     = np.arange(0, 101, 1.0)


# ── Controller 1: HVAC ───────────────────────────────────────────────────────

def build_hvac_controller():
    temperature   = ctrl.Antecedent(temp_uni,  'temperature')
    humidity      = ctrl.Antecedent(hum_uni,   'humidity')
    time_of_day   = ctrl.Antecedent(tod_uni,   'time_of_day')
    hvac_power    = ctrl.Consequent(hvac_uni,  'hvac_power')

    # Membership functions — temperature
    temperature['cold']       = fuzz.trapmf(temp_uni, [0,  0,  15, 18])
    temperature['cool']       = fuzz.trimf(temp_uni,  [15, 18, 21])
    temperature['comfortable']= fuzz.trimf(temp_uni,  [19, 21, 24])
    temperature['warm']       = fuzz.trimf(temp_uni,  [22, 25, 28])
    temperature['hot']        = fuzz.trapmf(temp_uni, [26, 30, 40, 40])

    # Membership functions — humidity
    humidity['dry']      = fuzz.trapmf(hum_uni, [0,  0,  30, 40])
    humidity['normal']   = fuzz.trimf(hum_uni,  [35, 50, 65])
    humidity['humid']    = fuzz.trapmf(hum_uni, [60, 70, 100, 100])

    # Membership functions — time_of_day
    time_of_day['night']     = fuzz.trapmf(tod_uni, [0,  0,  5,  7])
    time_of_day['morning']   = fuzz.trimf(tod_uni,  [6,  9,  12])
    time_of_day['afternoon'] = fuzz.trimf(tod_uni,  [11, 14, 17])
    time_of_day['evening']   = fuzz.trimf(tod_uni,  [16, 19, 22])
    time_of_day['late_night']= fuzz.trapmf(tod_uni, [21, 23, 24, 24])

    # Membership functions — hvac output
    hvac_power['strong_cool']  = fuzz.trapmf(hvac_uni, [-100, -100, -70, -50])
    hvac_power['mild_cool']    = fuzz.trimf(hvac_uni,  [-60,  -35,  -10])
    hvac_power['neutral']      = fuzz.trimf(hvac_uni,  [-15,    0,   15])
    hvac_power['mild_heat']    = fuzz.trimf(hvac_uni,  [10,    35,   60])
    hvac_power['strong_heat']  = fuzz.trapmf(hvac_uni, [50,   70, 100, 100])

    hvac_power.defuzzify_method = 'centroid'

    # Rule base
    rules = [
        # Temperature driven
        ctrl.Rule(temperature['hot']  & humidity['humid'],  hvac_power['strong_cool']),
        ctrl.Rule(temperature['hot']  & humidity['normal'], hvac_power['strong_cool']),
        ctrl.Rule(temperature['hot']  & humidity['dry'],    hvac_power['mild_cool']),
        ctrl.Rule(temperature['warm'] & humidity['humid'],  hvac_power['mild_cool']),
        ctrl.Rule(temperature['warm'] & humidity['normal'], hvac_power['neutral']),
        ctrl.Rule(temperature['warm'] & humidity['dry'],    hvac_power['neutral']),
        ctrl.Rule(temperature['comfortable'],               hvac_power['neutral']),
        ctrl.Rule(temperature['cool'] & humidity['humid'],  hvac_power['neutral']),
        ctrl.Rule(temperature['cool'] & humidity['normal'], hvac_power['mild_heat']),
        ctrl.Rule(temperature['cool'] & humidity['dry'],    hvac_power['mild_heat']),
        ctrl.Rule(temperature['cold'] & humidity['humid'],  hvac_power['mild_heat']),
        ctrl.Rule(temperature['cold'] & humidity['normal'], hvac_power['strong_heat']),
        ctrl.Rule(temperature['cold'] & humidity['dry'],    hvac_power['strong_heat']),
        # Time-of-day modulation (night → prefer warmer to assist sleep)
        ctrl.Rule(time_of_day['night'] & temperature['cool'],    hvac_power['mild_heat']),
        ctrl.Rule(time_of_day['night'] & temperature['cold'],    hvac_power['strong_heat']),
        ctrl.Rule(time_of_day['late_night'] & temperature['cool'], hvac_power['mild_heat']),
        # Afternoon peak heat
        ctrl.Rule(time_of_day['afternoon'] & temperature['warm'], hvac_power['mild_cool']),
    ]

    system  = ctrl.ControlSystem(rules)
    sim     = ctrl.ControlSystemSimulation(system)
    return sim, temperature, humidity, time_of_day, hvac_power


# ── Controller 2: Lighting ───────────────────────────────────────────────────

def build_lighting_controller():
    ambient_light = ctrl.Antecedent(light_uni, 'ambient_light')
    time_of_day   = ctrl.Antecedent(tod_uni,   'time_of_day')
    light_level   = ctrl.Consequent(ll_uni,    'light_level')

    ambient_light['dark']     = fuzz.trapmf(light_uni, [0,   0,  100, 200])
    ambient_light['dim']      = fuzz.trimf(light_uni,  [150, 300, 450])
    ambient_light['moderate'] = fuzz.trimf(light_uni,  [400, 550, 700])
    ambient_light['bright']   = fuzz.trapmf(light_uni, [650, 800, 1000, 1000])

    time_of_day['night']     = fuzz.trapmf(tod_uni, [0,  0,  5,  7])
    time_of_day['morning']   = fuzz.trimf(tod_uni,  [6,  9,  12])
    time_of_day['afternoon'] = fuzz.trimf(tod_uni,  [11, 14, 17])
    time_of_day['evening']   = fuzz.trimf(tod_uni,  [16, 19, 22])
    time_of_day['late_night']= fuzz.trapmf(tod_uni, [21, 23, 24, 24])

    light_level['off']         = fuzz.trapmf(ll_uni, [0,  0,  5,  10])
    light_level['dim']         = fuzz.trimf(ll_uni,  [5,  20, 35])
    light_level['medium']      = fuzz.trimf(ll_uni,  [30, 50, 70])
    light_level['bright']      = fuzz.trimf(ll_uni,  [65, 80, 90])
    light_level['full']        = fuzz.trapmf(ll_uni, [85, 95, 100, 100])

    light_level.defuzzify_method = 'centroid'

    rules = [
        ctrl.Rule(ambient_light['dark']     & time_of_day['night'],    light_level['dim']),
        ctrl.Rule(ambient_light['dark']     & time_of_day['late_night'],light_level['off']),
        ctrl.Rule(ambient_light['dark']     & time_of_day['morning'],  light_level['bright']),
        ctrl.Rule(ambient_light['dark']     & time_of_day['afternoon'],light_level['full']),
        ctrl.Rule(ambient_light['dark']     & time_of_day['evening'],  light_level['medium']),
        ctrl.Rule(ambient_light['dim']      & time_of_day['morning'],  light_level['medium']),
        ctrl.Rule(ambient_light['dim']      & time_of_day['afternoon'],light_level['medium']),
        ctrl.Rule(ambient_light['dim']      & time_of_day['evening'],  light_level['medium']),
        ctrl.Rule(ambient_light['dim']      & time_of_day['night'],    light_level['dim']),
        ctrl.Rule(ambient_light['moderate'] & time_of_day['morning'],  light_level['dim']),
        ctrl.Rule(ambient_light['moderate'] & time_of_day['afternoon'],light_level['off']),
        ctrl.Rule(ambient_light['moderate'] & time_of_day['evening'],  light_level['dim']),
        ctrl.Rule(ambient_light['moderate'] & time_of_day['night'],    light_level['off']),
        ctrl.Rule(ambient_light['bright'],                              light_level['off']),
    ]

    system = ctrl.ControlSystem(rules)
    sim    = ctrl.ControlSystemSimulation(system)
    return sim, ambient_light, time_of_day, light_level


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_membership_functions(temperature, humidity, time_of_day, hvac_power,
                               ambient_light, light_level):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        'FLC Membership Functions — Intelligent Assistive Care Environment\n'
        'HVAC Control (left) | Lighting Control (right)',
        fontsize=13, fontweight='bold'
    )

    vars_hvac    = [temperature, humidity, time_of_day, hvac_power]
    vars_light   = [ambient_light, time_of_day, light_level]
    colors       = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # HVAC inputs + output
    for ax, var in zip([axes[0,0], axes[1,0], axes[2,0]],
                       [temperature, hvac_power, time_of_day]):
        for i, (label, mf) in enumerate(var.terms.items()):
            ax.plot(var.universe, mf.mf, lw=2, label=label, color=colors[i % len(colors)])
        ax.set_title(f'HVAC: {var.label}')
        ax.set_xlabel(var.label)
        ax.set_ylabel('Membership')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    # Lighting inputs + output
    for ax, var in zip([axes[0,1], axes[1,1], axes[2,1]],
                       [ambient_light, time_of_day, light_level]):
        for i, (label, mf) in enumerate(var.terms.items()):
            ax.plot(var.universe, mf.mf, lw=2, label=label, color=colors[i % len(colors)])
        ax.set_title(f'Lighting: {var.label}')
        ax.set_xlabel(var.label)
        ax.set_ylabel('Membership')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task2_flc_membership_functions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_control_surface(hvac_sim, light_sim):
    """Generate 2D control surface plots (fix one input, sweep two)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'FLC Control Surfaces\n'
        'HVAC: Temperature × Humidity at fixed Time=14h  |  '
        'Lighting: Ambient Light × Time of Day',
        fontsize=12, fontweight='bold'
    )

    # HVAC surface: temperature × humidity (time fixed at 14h)
    temps  = np.linspace(0, 40, 40)
    hums   = np.linspace(0, 100, 40)
    T, H   = np.meshgrid(temps, hums)
    Z_hvac = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            try:
                hvac_sim.input['temperature']  = T[i, j]
                hvac_sim.input['humidity']     = H[i, j]
                hvac_sim.input['time_of_day']  = 14.0
                hvac_sim.compute()
                Z_hvac[i, j] = hvac_sim.output['hvac_power']
            except Exception:
                Z_hvac[i, j] = 0.0

    ax = axes[0]
    c  = ax.contourf(T, H, Z_hvac, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, label='HVAC Power (%)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Humidity (%)')
    ax.set_title('HVAC Control Surface (Time=14h)')
    ax.grid(True, alpha=0.2)

    # Lighting surface: ambient_light × time_of_day
    lights = np.linspace(0, 1000, 40)
    times  = np.linspace(0, 24,  40)
    L, TOD = np.meshgrid(lights, times)
    Z_ll   = np.zeros_like(L)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            try:
                light_sim.input['ambient_light'] = L[i, j]
                light_sim.input['time_of_day']   = TOD[i, j]
                light_sim.compute()
                Z_ll[i, j] = light_sim.output['light_level']
            except Exception:
                Z_ll[i, j] = 0.0

    ax = axes[1]
    c  = ax.contourf(L, TOD, Z_ll, levels=20, cmap='YlOrBr')
    plt.colorbar(c, ax=ax, label='Light Level (%)')
    ax.set_xlabel('Ambient Light (lux)')
    ax.set_ylabel('Time of Day (h)')
    ax.set_title('Lighting Control Surface')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task2_flc_control_surface.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_scenario_demo(hvac_sim, light_sim):
    """Demonstrate FLC output over 24h for a typical resident day."""
    hours = np.linspace(0, 23.75, 96)

    # Temperature profile: cool morning, warm afternoon, cool evening
    temp_profile = 18 + 6 * np.sin((hours - 6) * np.pi / 14)
    temp_profile = np.clip(temp_profile, 14, 30)

    hum_profile  = 55 + 10 * np.sin(hours * 2 * np.pi / 24)
    light_profile = 50 + 900 * np.clip(np.sin((hours - 6) * np.pi / 12), 0, 1)

    hvac_out   = []
    light_out  = []

    for h, t, hu, l in zip(hours, temp_profile, hum_profile, light_profile):
        try:
            hvac_sim.input['temperature']  = t
            hvac_sim.input['humidity']     = hu
            hvac_sim.input['time_of_day']  = h
            hvac_sim.compute()
            hvac_out.append(hvac_sim.output['hvac_power'])
        except Exception:
            hvac_out.append(0.0)
        try:
            light_sim.input['ambient_light'] = l
            light_sim.input['time_of_day']   = h
            light_sim.compute()
            light_out.append(light_sim.output['light_level'])
        except Exception:
            light_out.append(0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        'FLC Operational Scenario — 24h Assistive Care Flat\n'
        'Mamdani Fuzzy Inference Controller Response',
        fontsize=13, fontweight='bold'
    )

    axes[0].plot(hours, temp_profile, 'b-', lw=2, label='Temperature (°C)')
    axes[0].plot(hours, hum_profile,  'g--', lw=2, label='Humidity (%)')
    axes[0].set_ylabel('Sensor Readings')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Sensor Inputs')

    axes[1].plot(hours, hvac_out, 'r-', lw=2.5, label='HVAC Power')
    axes[1].axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)
    axes[1].fill_between(hours, hvac_out, 0,
                         where=np.array(hvac_out) > 0, alpha=0.2, color='red', label='Heating')
    axes[1].fill_between(hours, hvac_out, 0,
                         where=np.array(hvac_out) < 0, alpha=0.2, color='blue', label='Cooling')
    axes[1].set_ylabel('HVAC Power (%)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('HVAC Controller Output')

    axes[2].plot(hours, light_out, 'orange', lw=2.5, label='Light Level (%)')
    axes[2].fill_between(hours, light_out, 0, alpha=0.3, color='orange')
    axes[2].set_ylabel('Light Level (%)')
    axes[2].set_xlabel('Time of Day (h)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Lighting Controller Output')
    axes[2].set_xlim(0, 24)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task2_flc_scenario.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("\n[Task 2, Part 1] Fuzzy Logic Controller Design")

    hvac_sim, temperature, humidity, time_of_day, hvac_power = build_hvac_controller()
    light_sim, ambient_light, tod2, light_level = build_lighting_controller()

    print("  HVAC Controller: 3 inputs × 5/3/5 MFs → 1 output (5 MFs), 17 rules")
    print("  Lighting Controller: 2 inputs × 4/5 MFs → 1 output (5 MFs), 14 rules")
    print("  Defuzzification: Centroid (Centre of Gravity)")

    # Quick test
    hvac_sim.input['temperature']  = 30.0
    hvac_sim.input['humidity']     = 70.0
    hvac_sim.input['time_of_day']  = 14.0
    hvac_sim.compute()
    print(f"\n  Test (Temp=30°C, Hum=70%, Time=14h): HVAC = {hvac_sim.output['hvac_power']:.1f}%")

    light_sim.input['ambient_light'] = 100.0
    light_sim.input['time_of_day']   = 20.0
    light_sim.compute()
    print(f"  Test (Light=100 lux, Time=20h): Light Level = {light_sim.output['light_level']:.1f}%")

    print("\n  Generating plots...")
    plot_membership_functions(temperature, humidity, time_of_day, hvac_power,
                              ambient_light, light_level)
    plot_control_surface(hvac_sim, light_sim)
    plot_scenario_demo(hvac_sim, light_sim)

    return hvac_sim, light_sim


if __name__ == '__main__':
    main()
