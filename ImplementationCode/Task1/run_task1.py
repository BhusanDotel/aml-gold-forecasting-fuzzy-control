"""
Task 1 Runner — Probabilistic Machine Learning Approaches for Gold Market Forecasting
and Financial Text Mining.

Runs all three components sequentially and prints a consolidated summary.
"""

import os, sys, time
import numpy as np

# Ensure local imports work regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))


def run_all():
    print("=" * 65)
    print("  TASK 1: Probabilistic ML for Gold Market Forecasting")
    print("  Module: 7078CEM Advanced Machine Learning")
    print("=" * 65)

    results = {}

    # ── Component 1: GP Regression & Classification ────────────────────
    t0 = time.time()
    print("\n[1/3] Gaussian Process Regression & Classification")
    from gp_regression_classification import main as gpr_main
    gpr, gpc, reg_metrics, cls_metrics = gpr_main()
    # price-level metrics printed inline by gp_regression_classification
    results['GPR'] = reg_metrics
    results['GPC'] = cls_metrics
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Component 2: Bayesian Network ──────────────────────────────────
    t0 = time.time()
    print("\n[2/3] Bayesian Network")
    from bayesian_network import main as bn_main
    model_bn, infer, bn_metrics = bn_main()
    results['BN'] = bn_metrics
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Component 3: LDA Text Mining ───────────────────────────────────
    t0 = time.time()
    print("\n[3/3] LDA Text Mining")
    from lda_text_mining import main as lda_main
    lda_model, lda_metrics = lda_main()
    results['LDA'] = lda_metrics
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  GPR  MSE  : {results['GPR']['MSE']:.6f}")
    print(f"  GPR  MAE  : {results['GPR']['MAE']:.6f}")
    print(f"  GPR  R²   : {results['GPR']['R2']:.4f}")
    print(f"  GPC  Acc  : {results['GPC']['Accuracy']:.4f}")
    print(f"  BN   Acc  : {results['BN']['Accuracy']:.4f}")
    print(f"  LDA  TopicAcc: {results['LDA']['Topic_Assignment_Accuracy']:.4f}")

    assets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
    saved = sorted(f for f in os.listdir(assets_dir) if f.startswith('task1_'))
    print(f"\n  Saved {len(saved)} asset(s) to assets/:")
    for f in sorted(saved):
        print(f"    {f}")
    print("=" * 65)
    return results


if __name__ == '__main__':
    run_all()
