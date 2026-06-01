"""
Task 2 Runner — Evolutionary and Fuzzy Systems.

  Part 1: FLC Design and Implementation
  Part 2: GA Optimisation of FLC Membership Functions
  Part 3: CEC'2005 Benchmark Comparison (GA vs PSO)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def run_all():
    print("=" * 65)
    print("  TASK 2: Evolutionary and Fuzzy Systems")
    print("  Module: 7078CEM Advanced Machine Learning")
    print("=" * 65)

    # ── Part 1: FLC ────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1/3] FLC Design and Implementation")
    from flc_design import main as flc_main
    hvac_sim, light_sim = flc_main()
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Part 2: GA Optimisation ────────────────────────────────────────
    t0 = time.time()
    print("\n[2/3] GA Optimisation of FLC Membership Functions")
    from ga_optimization import main as ga_main
    best_chrom, best_rmse = ga_main()
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Part 3: CEC'2005 ───────────────────────────────────────────────
    t0 = time.time()
    print("\n[3/3] CEC'2005 Benchmark Comparison")
    from cec2005_benchmark import main as cec_main
    cec_results = cec_main()
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  FLC: 2 controllers (HVAC + Lighting), Mamdani model")
    print(f"       HVAC: 3 inputs, 17 rules | Lighting: 2 inputs, 14 rules")
    print(f"  GA-FLC Optimised RMSE: {best_rmse:.4f}")
    print(f"  CEC'2005 Results (mean error, 15 runs):")
    for fn, fn_res in cec_results.items():
        for D, algs in fn_res.items():
            for alg, stats in algs.items():
                print(f"    {fn[:20]:<20} D={D:<3} {alg:<4}  mean={stats['mean']:.4f}  "
                      f"std={stats['std']:.4f}")

    assets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
    saved = [f for f in os.listdir(assets_dir) if f.startswith('task2_')]
    print(f"\n  Saved {len(saved)} asset(s) to assets/:")
    for f in sorted(saved):
        print(f"    {f}")
    print("=" * 65)


if __name__ == '__main__':
    run_all()
