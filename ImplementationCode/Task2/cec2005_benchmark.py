"""
Task 2, Part 3 — CEC'2005 Benchmark: Compare Optimisation Techniques.

Functions chosen:
  F1 — Shifted Sphere Function        (unimodal, separable, global min = -450)
  F6 — Shifted Rotated Ackley's Function  (multimodal, non-separable, global min = -140)

Algorithms:
  1. Genetic Algorithm (GA) — real-valued, tournament selection, BLX-α crossover
  2. Particle Swarm Optimisation (PSO) — global topology

Dimensions: D = 2 and D = 10
Runs: 15 per combination
Max function evaluations: 10 000 × D (CEC'2005 standard)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

# ── CEC'2005 function definitions ─────────────────────────────────────────────

def make_shift_vector(D, seed, lb, ub):
    """Generate the shifted global optimum o."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lb + 0.1 * (ub - lb), ub - 0.1 * (ub - lb), D)


def make_rotation_matrix(D, seed):
    """Orthonormal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed + 100)
    A   = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    return Q


def f1_shifted_sphere(x, o):
    """F1: Shifted Sphere — f(x) = sum((x-o)^2) - 450"""
    z = x - o
    return np.dot(z, z) - 450.0


def f6_shifted_rotated_ackley(x, o, M):
    """F6: Shifted Rotated Ackley's — global min = -140 at x=o"""
    z = M @ (x - o)
    n = len(z)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sqrt(np.sum(z ** 2) / n)
    s2 = np.sum(np.cos(c * z)) / n
    return -a * np.exp(-b * s1) - np.exp(s2) + a + np.e - 140.0


FUNCTIONS = {
    'F1 Shifted Sphere': {
        'fn': f1_shifted_sphere,
        'opt': -450.0,
        'lb': -100.0,
        'ub':  100.0,
        'needs_rotation': False,
    },
    'F6 Shifted Rotated Ackley': {
        'fn': f6_shifted_rotated_ackley,
        'opt': -140.0,
        'lb': -32.0,
        'ub':  32.0,
        'needs_rotation': True,
    },
}


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def ga_optimize(obj, lb, ub, D, max_evals=None, seed=0,
                pop_size=None, alpha=0.5):
    """Real-valued GA with BLX-α crossover, tournament selection, Gaussian mutation."""
    if max_evals is None:
        max_evals = 10_000 * D
    if pop_size is None:
        pop_size = max(20, 10 * D)

    rng  = np.random.default_rng(seed)
    pop  = rng.uniform(lb, ub, (pop_size, D))
    fit  = np.array([obj(ind) for ind in pop])
    evals = pop_size

    best_val   = fit.min()
    best_curve = [best_val]

    p_mut = 1.0 / D
    sigma = (ub - lb) * 0.1

    while evals < max_evals:
        new_pop = []
        # Elitism
        new_pop.append(pop[fit.argmin()].copy())

        while len(new_pop) < pop_size:
            # Tournament selection (k=3)
            i1 = rng.integers(0, pop_size, 3); p1 = pop[i1[fit[i1].argmin()]].copy()
            i2 = rng.integers(0, pop_size, 3); p2 = pop[i2[fit[i2].argmin()]].copy()

            # BLX-α crossover
            d   = np.abs(p1 - p2)
            lo_ = np.minimum(p1, p2) - alpha * d
            hi_ = np.maximum(p1, p2) + alpha * d
            c1  = rng.uniform(lo_, hi_)
            c2  = rng.uniform(lo_, hi_)

            # Gaussian mutation
            mask = rng.random(D) < p_mut
            c1[mask] += rng.normal(0, sigma, mask.sum())
            c2[mask] += rng.normal(0, sigma, mask.sum())

            c1 = np.clip(c1, lb, ub)
            c2 = np.clip(c2, lb, ub)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:pop_size])
        fit = np.array([obj(ind) for ind in pop])
        evals += pop_size

        cur_best = fit.min()
        if cur_best < best_val:
            best_val = cur_best
        best_curve.append(best_val)

    return best_val, np.array(best_curve)


# ── Particle Swarm Optimisation ───────────────────────────────────────────────

def pso_optimize(obj, lb, ub, D, max_evals=None, seed=0,
                 n_particles=None, w=0.729, c1=1.494, c2=1.494):
    """Standard PSO — global best topology."""
    if max_evals is None:
        max_evals = 10_000 * D
    if n_particles is None:
        n_particles = max(20, 10 * D)

    rng = np.random.default_rng(seed)
    pos = rng.uniform(lb, ub, (n_particles, D))
    vel = rng.uniform(-(ub - lb) * 0.1, (ub - lb) * 0.1, (n_particles, D))
    fit = np.array([obj(p) for p in pos])
    evals = n_particles

    pbest_pos = pos.copy()
    pbest_fit = fit.copy()

    gbest_idx = pbest_fit.argmin()
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_fit[gbest_idx]

    best_curve = [gbest_val]

    while evals < max_evals:
        r1 = rng.random((n_particles, D))
        r2 = rng.random((n_particles, D))
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = np.clip(pos + vel, lb, ub)
        fit = np.array([obj(p) for p in pos])
        evals += n_particles

        improve = fit < pbest_fit
        pbest_pos[improve] = pos[improve].copy()
        pbest_fit[improve] = fit[improve]

        gi = pbest_fit.argmin()
        if pbest_fit[gi] < gbest_val:
            gbest_val = pbest_fit[gi]
            gbest_pos = pbest_pos[gi].copy()

        best_curve.append(gbest_val)

    return gbest_val, np.array(best_curve)


# ── Runner ────────────────────────────────────────────────────────────────────

N_RUNS = 15

def run_experiments():
    results = {}
    curves  = {}

    for fn_name, fn_cfg in FUNCTIONS.items():
        results[fn_name] = {}
        curves[fn_name]  = {}
        for D in [2, 10]:
            results[fn_name][D] = {}
            curves[fn_name][D]  = {}
            max_evals = 10_000 * D

            # Pre-generate one shift vector and rotation matrix per (fn, D)
            seed_base = 42
            o = make_shift_vector(D, seed_base, fn_cfg['lb'], fn_cfg['ub'])
            if fn_cfg['needs_rotation']:
                M = make_rotation_matrix(D, seed_base)
                def obj(x, _o=o, _M=M, _cfg=fn_cfg): return _cfg['fn'](x, _o, _M)
            else:
                def obj(x, _o=o, _cfg=fn_cfg): return _cfg['fn'](x, _o)

            for alg_name, optimizer in [('GA', ga_optimize), ('PSO', pso_optimize)]:
                best_vals = []
                alg_curves = []
                for run in range(N_RUNS):
                    best_val, curve = optimizer(
                        obj, fn_cfg['lb'], fn_cfg['ub'], D,
                        max_evals=max_evals, seed=run * 13 + D * 7
                    )
                    best_vals.append(best_val)
                    alg_curves.append(curve)

                best_vals = np.array(best_vals)
                opt       = fn_cfg['opt']
                errors    = best_vals - opt

                results[fn_name][D][alg_name] = {
                    'mean':  errors.mean(),
                    'std':   errors.std(),
                    'best':  errors.min(),
                    'worst': errors.max(),
                    'median':np.median(errors),
                }
                curves[fn_name][D][alg_name] = alg_curves

    return results, curves


def print_results(results):
    print("\n" + "=" * 70)
    print("CEC'2005 Benchmark Results  (error = best_found − global_optimum)")
    print("=" * 70)
    for fn_name, fn_res in results.items():
        for D, d_res in fn_res.items():
            print(f"\n  {fn_name}  (D={D})")
            print(f"  {'Algorithm':<8}  {'Mean':>12}  {'Std':>12}  {'Best':>12}  {'Worst':>12}")
            print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
            for alg, stats in d_res.items():
                print(f"  {alg:<8}  {stats['mean']:>12.4f}  {stats['std']:>12.4f}  "
                      f"{stats['best']:>12.4f}  {stats['worst']:>12.4f}")


def plot_results(results, curves):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "CEC'2005 Benchmark: GA vs PSO  (15 runs each)\n"
        "F1: Shifted Sphere  |  F6: Shifted Rotated Ackley's",
        fontsize=13, fontweight='bold'
    )

    fn_names = list(FUNCTIONS.keys())
    dims     = [2, 10]
    alg_colors = {'GA': '#e74c3c', 'PSO': '#3498db'}

    for row, fn_name in enumerate(fn_names):
        for col_offset, D in enumerate(dims):
            # Convergence plot
            ax_conv = axes[row, col_offset * 2]
            for alg, color in alg_colors.items():
                run_curves = curves[fn_name][D][alg]
                # Normalise curve lengths and compute mean convergence
                min_len = min(len(c) for c in run_curves)
                mat = np.array([c[:min_len] for c in run_curves])
                mean_curve = mat.mean(axis=0)
                opt = FUNCTIONS[fn_name]['opt']
                error_curve = np.maximum(mean_curve - opt, 1e-10)
                ax_conv.semilogy(error_curve, color=color, lw=2, label=alg)
                for c in mat:
                    ec = np.maximum(c[:min_len] - opt, 1e-10)
                    ax_conv.semilogy(ec, color=color, lw=0.4, alpha=0.2)
            ax_conv.set_title(f'{fn_name.split(" ")[0]}{fn_name.split(" ")[1]}  D={D}\nConvergence')
            ax_conv.set_xlabel('Iteration')
            ax_conv.set_ylabel('Error (log scale)')
            ax_conv.legend(fontsize=8)
            ax_conv.grid(True, which='both', alpha=0.3)

            # Box plot of final errors
            ax_box = axes[row, col_offset * 2 + 1]
            opt  = FUNCTIONS[fn_name]['opt']
            data = []
            labels = []
            for alg in ['GA', 'PSO']:
                run_curves = curves[fn_name][D][alg]
                finals = np.array([c[-1] for c in run_curves]) - opt
                data.append(np.maximum(finals, 1e-10))
                labels.append(alg)
            bp = ax_box.boxplot(data, labels=labels, patch_artist=True,
                                medianprops=dict(color='black', lw=2))
            for patch, alg in zip(bp['boxes'], ['GA', 'PSO']):
                patch.set_facecolor(alg_colors[alg])
                patch.set_alpha(0.7)
            ax_box.set_yscale('log')
            ax_box.set_title(f'{fn_name.split(" ")[0]}{fn_name.split(" ")[1]}  D={D}\nFinal Error Distribution')
            ax_box.set_ylabel('Final Error (log scale)')
            ax_box.grid(True, axis='y', which='both', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task2_cec2005_benchmark.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("\n[Task 2, Part 3] CEC'2005 Benchmark Comparison")
    print(f"  Functions: F1 (Shifted Sphere), F6 (Shifted Rotated Ackley's)")
    print(f"  Algorithms: GA (BLX-α crossover) and PSO (global topology)")
    print(f"  Dimensions: D=2, D=10")
    print(f"  Runs per combination: {N_RUNS}")
    print(f"  Max evaluations: 10,000 × D\n")

    print("  Running experiments (this may take ~2-3 minutes)...")
    results, curves = run_experiments()
    print_results(results)
    plot_results(results, curves)
    return results


if __name__ == '__main__':
    main()
