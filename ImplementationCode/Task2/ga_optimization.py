"""
Task 2, Part 2 — Genetic Algorithm to optimise the FLC membership functions.

Uses the same HVAC FLC from Part 1.

Chromosome encodes the centres of all triangular MFs (continuous parameters):
  temperature  : 5 terms × 2 params (centre, half-width)  = 10 genes
  humidity     : 3 terms × 2 params                        =  6 genes
  hvac_power   : 5 terms × 2 params                        = 10 genes
  Total: 26 genes per chromosome (real-valued)

Fitness: RMSE on a synthetic input–output dataset derived from expert rules.
Operators: tournament selection, uniform crossover, Gaussian mutation.

Note: runs a lightweight version of the controller for speed, using direct
      fuzz.defuzz calls rather than the skfuzzy control system overhead.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import skfuzzy as fuzz

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

# ── Universe (coarse for speed) ───────────────────────────────────────────────
TEMP_U  = np.arange(0, 41, 1.0)
HUM_U   = np.arange(0, 101, 2.0)
HVAC_U  = np.arange(-100, 101, 2.0)

# ── Chromosome structure ──────────────────────────────────────────────────────
# Gene layout: [t_c1..t_c5, t_w1..t_w5, h_c1..h_c3, h_w1..h_w3,
#               o_c1..o_c5, o_w1..o_w5]
# Centres (c) normalised to [0,1], widths (w) in [0.05, 0.5]

N_GENES  = 26   # 10 (temp) + 6 (hum) + 10 (hvac)
POP_SIZE = 40
N_GEN    = 60
P_CROSS  = 0.85
P_MUT    = 0.08
TOURN_K  = 4


# ── Ground-truth dataset (expert rules) ──────────────────────────────────────

def generate_dataset(n=400, seed=7):
    """Expert rule dataset: (temp, hum) → expected hvac output."""
    rng = np.random.default_rng(seed)
    temps = rng.uniform(0, 40, n)
    hums  = rng.uniform(0, 100, n)
    hvac  = np.zeros(n)

    for i, (t, h) in enumerate(zip(temps, hums)):
        if t >= 30 and h >= 60:
            hvac[i] = -80
        elif t >= 30:
            hvac[i] = -60
        elif t >= 26:
            hvac[i] = -30 + (h - 50) * 0.5
        elif 19 <= t <= 24:
            hvac[i] = 0 + (h - 50) * 0.2
        elif t <= 15 and h >= 60:
            hvac[i] = 40
        elif t <= 15:
            hvac[i] = 65
        else:
            hvac[i] = (21 - t) * 12 - (h - 50) * 0.3
    hvac = np.clip(hvac, -100, 100)
    return temps, hums, hvac


TEMPS_GT, HUMS_GT, HVAC_GT = generate_dataset()


# ── Chromosome decode → MF parameters ────────────────────────────────────────

def decode(chrom):
    """Decode chromosome into sorted centre arrays and width arrays."""
    # temperature: 5 centres + 5 widths
    tc = np.sort(chrom[0:5]) * 40.0          # denorm to [0,40]
    tw = chrom[5:10] * 8.0 + 1.5             # widths in [1.5, 9.5]
    # humidity: 3 centres + 3 widths
    hc = np.sort(chrom[10:13]) * 100.0       # [0,100]
    hw = chrom[13:16] * 30.0 + 5.0           # [5,35]
    # hvac: 5 centres + 5 widths
    oc = np.sort(chrom[16:21]) * 200.0 - 100 # [-100,100]
    ow = chrom[21:26] * 40.0 + 10.0          # [10,50]
    return tc, tw, hc, hw, oc, ow


def evaluate_flc(chrom):
    """Evaluate chromosome: compute RMSE on the expert dataset."""
    tc, tw, hc, hw, oc, ow = decode(chrom)
    errors = []

    for temp_v, hum_v, target in zip(TEMPS_GT, HUMS_GT, HVAC_GT):
        # Temperature MFs (triangular)
        temp_mf = np.array([
            fuzz.trimf(TEMP_U, [tc[i] - tw[i], tc[i], tc[i] + tw[i]])
            for i in range(5)
        ])
        temp_deg = np.array([fuzz.interp_membership(TEMP_U, mf, temp_v) for mf in temp_mf])

        # Humidity MFs
        hum_mf = np.array([
            fuzz.trimf(HUM_U, [hc[i] - hw[i], hc[i], hc[i] + hw[i]])
            for i in range(3)
        ])
        hum_deg = np.array([fuzz.interp_membership(HUM_U, mf, hum_v) for mf in hum_mf])

        # HVAC output MFs
        hvac_mf = np.array([
            fuzz.trimf(HVAC_U, [oc[i] - ow[i], oc[i], oc[i] + ow[i]])
            for i in range(5)
        ])

        # Simplified rule base: 5 temp × 3 hum combinations → 5 HVAC terms
        # Firing strengths via min (Mamdani AND)
        rule_strengths = np.zeros(5)
        combos = [(4, 2, 0), (4, 1, 0), (3, 2, 1), (2, 1, 2), (1, 0, 3), (0, 0, 4), (0, 1, 4)]
        for t_idx, h_idx, o_idx in combos:
            strength = min(temp_deg[t_idx], hum_deg[h_idx])
            rule_strengths[o_idx] = max(rule_strengths[o_idx], strength)

        # Aggregate (max) and defuzzify (centroid)
        agg = np.zeros_like(HVAC_U)
        for o_idx in range(5):
            agg = np.fmax(agg, np.fmin(rule_strengths[o_idx], hvac_mf[o_idx]))

        if agg.sum() < 1e-6:
            pred = 0.0
        else:
            pred = fuzz.defuzz(HVAC_U, agg, 'centroid')

        errors.append((pred - target) ** 2)

    return np.sqrt(np.mean(errors))


# ── GA operators ──────────────────────────────────────────────────────────────

def init_population(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (POP_SIZE, N_GENES))


def tournament_select(pop, fitnesses, k=TOURN_K, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(pop), k)
    best = idx[fitnesses[idx].argmin()]
    return pop[best].copy()


def uniform_crossover(p1, p2, rng):
    mask = rng.random(N_GENES) < 0.5
    child1 = np.where(mask, p1, p2)
    child2 = np.where(mask, p2, p1)
    return child1, child2


def gaussian_mutation(chrom, p_mut, sigma=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(N_GENES) < p_mut
    chrom = chrom.copy()
    chrom[mask] += rng.normal(0, sigma, mask.sum())
    return np.clip(chrom, 0, 1)


def run_ga(seed=42):
    rng = np.random.default_rng(seed)
    pop = init_population(seed=seed)

    fitnesses = np.array([evaluate_flc(c) for c in pop])
    best_idx  = fitnesses.argmin()
    best_fit  = fitnesses[best_idx]
    best_chrom = pop[best_idx].copy()

    history_best = [best_fit]
    history_mean = [fitnesses.mean()]

    print(f"  Gen 0  — Best RMSE: {best_fit:.4f}  Mean: {fitnesses.mean():.4f}")

    for gen in range(1, N_GEN + 1):
        new_pop = []
        # Elitism: carry over best individual unchanged
        new_pop.append(best_chrom.copy())

        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fitnesses, rng=rng)
            p2 = tournament_select(pop, fitnesses, rng=rng)
            if rng.random() < P_CROSS:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = gaussian_mutation(c1, P_MUT, rng=rng)
            c2 = gaussian_mutation(c2, P_MUT, rng=rng)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:POP_SIZE])
        fitnesses = np.array([evaluate_flc(c) for c in pop])

        gen_best_idx = fitnesses.argmin()
        if fitnesses[gen_best_idx] < best_fit:
            best_fit   = fitnesses[gen_best_idx]
            best_chrom = pop[gen_best_idx].copy()

        history_best.append(best_fit)
        history_mean.append(fitnesses.mean())

        if gen % 10 == 0:
            print(f"  Gen {gen:3d} — Best RMSE: {best_fit:.4f}  Mean: {fitnesses.mean():.4f}")

    return best_chrom, best_fit, history_best, history_mean


# ── Compare Sugeno vs Mamdani description ────────────────────────────────────

def describe_sugeno_alternative():
    """Explain how the GA changes for a Sugeno model (Part 2 requirement)."""
    print("\n  --- Sugeno (TSK) Alternative ---")
    print("  In a Sugeno model, output MFs are replaced by crisp constants or")
    print("  linear functions: y_k = a0_k + a1_k*temp + a2_k*hum.")
    print("  GA chromosome change:")
    print("    • Remove HVAC output MF centre/width genes (10 genes).")
    print("    • Add Sugeno consequent coefficients: 5 rules × 3 params = 15 genes.")
    print("    • Total chromosome: 10 + 6 + 15 = 31 genes.")
    print("    • Defuzzification: weighted-average (exact, no integration needed).")
    print("    • Fitness evaluation becomes faster → GA converges in fewer generations.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ga_results(history_best, history_mean, initial_rmse, final_rmse,
                    best_chrom, initial_chrom=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Genetic Algorithm — Optimisation of FLC Membership Functions\n'
        f'Initial RMSE: {initial_rmse:.3f}  →  Optimised RMSE: {final_rmse:.3f}',
        fontsize=12, fontweight='bold'
    )

    # 1 Convergence
    ax = axes[0]
    gens = range(len(history_best))
    ax.plot(gens, history_best, 'b-', lw=2, label='Best RMSE')
    ax.plot(gens, history_mean, 'r--', lw=1.5, label='Mean RMSE', alpha=0.7)
    ax.fill_between(gens, history_best, history_mean, alpha=0.15, color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('RMSE')
    ax.set_title('GA Convergence Curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2 Before vs after temperature MFs
    ax = axes[1]
    if initial_chrom is None:
        initial_chrom = np.array([0.2, 0.35, 0.5, 0.65, 0.8,
                                   0.1]*5 + [0.25, 0.5, 0.75, 0.1]*3 +
                                   [0.1, 0.3, 0.5, 0.7, 0.9, 0.1]*5)
        initial_chrom = initial_chrom[:N_GENES]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for chrom, ls, lbl in [(initial_chrom, '--', 'Before'), (best_chrom, '-', 'After')]:
        tc, tw, _, _, _, _ = decode(chrom)
        for i in range(5):
            mf = fuzz.trimf(TEMP_U, [tc[i] - tw[i], tc[i], tc[i] + tw[i]])
            ax.plot(TEMP_U, mf, lw=1.8, ls=ls, color=colors[i],
                    alpha=0.7, label=f'MF{i+1} {lbl}' if i == 0 else None)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], ls='--', color='grey', lw=1.5, label='Before GA'),
               Line2D([0], [0], ls='-',  color='grey', lw=1.5, label='After GA')]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Temperature MFs: Before vs After GA')
    ax.grid(True, alpha=0.3)

    # 3 Prediction vs target scatter (optimised)
    ax = axes[2]
    preds = []
    for t, h, target in zip(TEMPS_GT[:150], HUMS_GT[:150], HVAC_GT[:150]):
        tc, tw, hc, hw, oc, ow = decode(best_chrom)
        temp_mf = [fuzz.trimf(TEMP_U, [tc[i]-tw[i], tc[i], tc[i]+tw[i]]) for i in range(5)]
        hum_mf  = [fuzz.trimf(HUM_U,  [hc[i]-hw[i], hc[i], hc[i]+hw[i]]) for i in range(3)]
        hvac_mf = [fuzz.trimf(HVAC_U, [oc[i]-ow[i], oc[i], oc[i]+ow[i]]) for i in range(5)]
        td = [fuzz.interp_membership(TEMP_U, mf, t)  for mf in temp_mf]
        hd = [fuzz.interp_membership(HUM_U,  mf, h)  for mf in hum_mf]
        rs = np.zeros(5)
        for t_i, h_i, o_i in [(4,2,0),(4,1,0),(3,2,1),(2,1,2),(1,0,3),(0,0,4),(0,1,4)]:
            s = min(td[t_i], hd[h_i]); rs[o_i] = max(rs[o_i], s)
        agg = np.zeros_like(HVAC_U)
        for o_i in range(5):
            agg = np.fmax(agg, np.fmin(rs[o_i], hvac_mf[o_i]))
        pred = fuzz.defuzz(HVAC_U, agg, 'centroid') if agg.sum() > 1e-6 else 0.0
        preds.append(pred)
    preds = np.array(preds)
    targets = HVAC_GT[:150]
    ax.scatter(targets, preds, alpha=0.5, s=15, color='steelblue', edgecolors='none')
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')
    ax.set_xlabel('Target HVAC Output')
    ax.set_ylabel('Predicted HVAC Output')
    ax.set_title(f'GA-Optimised FLC: Pred vs Target (RMSE={final_rmse:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task2_ga_optimisation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("\n[Task 2, Part 2] Genetic Algorithm — FLC Optimisation")
    print(f"  Population: {POP_SIZE}  |  Generations: {N_GEN}")
    print(f"  Crossover: uniform (p={P_CROSS})  |  Mutation: Gaussian (p={P_MUT})")
    print(f"  Selection: Tournament (k={TOURN_K})  |  Elitism: 1 individual")
    print(f"  Chromosome length: {N_GENES} genes (centres + widths of all MFs)")
    print(f"  Fitness: RMSE on {len(TEMPS_GT)}-point expert dataset")

    # Evaluate initial random chromosome
    rng = np.random.default_rng(0)
    initial_chrom = rng.uniform(0, 1, N_GENES)
    initial_rmse  = evaluate_flc(initial_chrom)
    print(f"\n  Initial random chromosome RMSE: {initial_rmse:.4f}")

    print("\n  Running GA...")
    best_chrom, best_fit, hist_best, hist_mean = run_ga()

    print(f"\n  Final best RMSE: {best_fit:.4f}")
    print(f"  Improvement: {100*(initial_rmse - best_fit)/initial_rmse:.1f}%")

    describe_sugeno_alternative()
    plot_ga_results(hist_best, hist_mean, initial_rmse, best_fit,
                    best_chrom, initial_chrom)

    return best_chrom, best_fit


if __name__ == '__main__':
    main()
