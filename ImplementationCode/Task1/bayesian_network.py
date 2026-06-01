"""
Bayesian Network for XAUUSD M15 market structure modelling.

10 discrete variables:
  Open_cat, High_cat, Low_cat, Close_cat   — price level categories (0=Low, 1=Mid, 2=High)
  Volume_cat                               — volume (0=Low, 1=Mid, 2=High)
  RSI_cat                                  — RSI zone (0=Oversold, 1=Neutral, 2=Overbought)
  Trend_cat                                — price vs SMA-14 (0=Below, 1=Above)
  Session                                  — trading session (0=Asian, 1=European, 2=American)
  Volatility_cat                           — ATR quartile (0=Low, 1=High)
  Direction                                — next-bar direction (0=Down, 1=Up)

Structure: manually specified DAG that encodes domain knowledge about how
macro-session context flows into price behaviour and direction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
try:
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator as _Estimator
except ImportError:
    try:
        from pgmpy.parameter_estimator import DiscreteMLE as _Estimator
    except ImportError:
        _Estimator = None
from pgmpy.inference import VariableElimination

from data_preprocessing import load_data, add_technical_indicators, discretize_for_bn, BN_VARS

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

# ── DAG structure (domain-knowledge DAG) ─────────────────────────────────────
#
#  Session → RSI_cat
#  Session → Volume_cat
#  Session → Volatility_cat
#  Open_cat → High_cat
#  Open_cat → Low_cat
#  Open_cat → Close_cat
#  High_cat → Close_cat
#  Low_cat  → Close_cat
#  Close_cat → RSI_cat
#  Close_cat → Trend_cat
#  Trend_cat → Direction
#  RSI_cat   → Direction
#  Volume_cat → Direction
#  Volatility_cat → Direction
#
EDGES = [
    ('Session',       'RSI_cat'),
    ('Session',       'Volume_cat'),
    ('Session',       'Volatility_cat'),
    ('Open_cat',      'High_cat'),
    ('Open_cat',      'Low_cat'),
    ('Open_cat',      'Close_cat'),
    ('High_cat',      'Close_cat'),
    ('Low_cat',       'Close_cat'),
    ('Close_cat',     'RSI_cat'),
    ('Close_cat',     'Trend_cat'),
    ('Trend_cat',     'Direction'),
    ('RSI_cat',       'Direction'),
    ('Volume_cat',    'Direction'),
    ('Volatility_cat','Direction'),
]

CARD = {
    'Open_cat': 3, 'High_cat': 3, 'Low_cat': 3, 'Close_cat': 3,
    'Volume_cat': 3, 'RSI_cat': 3,
    'Trend_cat': 2, 'Session': 3, 'Volatility_cat': 2, 'Direction': 2
}


def build_and_fit_bn(df_disc):
    """Build the BN, fit CPDs via Bayesian estimation, return model + inference engine."""
    model = BayesianNetwork(EDGES)

    if _Estimator is not None:
        model.fit(df_disc, estimator=_Estimator())
    else:
        model.fit(df_disc)

    assert model.check_model(), "BN model check failed"
    infer = VariableElimination(model)
    return model, infer


def evaluate_bn(model, infer, df_disc, n_eval=500):
    """Evaluate Direction prediction accuracy via VE on held-out samples."""
    df_test = df_disc.iloc[-n_eval:].reset_index(drop=True)
    evidence_vars = [v for v in BN_VARS if v != 'Direction']

    correct = 0
    for _, row in df_test.iterrows():
        ev = {v: int(row[v]) for v in evidence_vars}
        try:
            q = infer.query(['Direction'], evidence=ev, show_progress=False)
            pred = int(q.values.argmax())
        except Exception:
            pred = 1  # default: predict up
        correct += int(pred == int(row['Direction']))

    acc = correct / n_eval
    print("\n" + "=" * 50)
    print("Bayesian Network Results")
    print("=" * 50)
    print(f"  Nodes : {len(model.nodes())}")
    print(f"  Edges : {len(model.edges())}")
    print(f"  Direction prediction accuracy (VE): {acc:.4f}")
    return acc


def print_cpds(model):
    """Print selected CPDs."""
    print("\n  Selected CPDs:")
    for node in ['Direction', 'RSI_cat']:
        cpd = model.get_cpds(node)
        print(f"\n  CPD({node}):")
        print(f"    Variables: {cpd.variables}")
        print(f"    Values shape: {cpd.get_values().shape}")
        print(f"    Values:\n{np.round(cpd.get_values(), 3)}")


def plot_bn(model, df_disc):
    """Plot BN graph and marginal distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Bayesian Network — XAUUSD M15 Market Structure\n'
        '(10 Variables, Domain-Knowledge DAG)',
        fontsize=13, fontweight='bold'
    )

    # 1 – Network graph via networkx
    import networkx as nx
    ax = axes[0, 0]
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, seed=42, k=2.0)
    node_colors = []
    for n in G.nodes():
        if n == 'Direction':
            node_colors.append('#e74c3c')
        elif n == 'Session':
            node_colors.append('#3498db')
        elif 'cat' in n or n in ['Trend_cat']:
            node_colors.append('#2ecc71')
        else:
            node_colors.append('#95a5a6')
    nx.draw_networkx(G, pos, ax=ax, node_color=node_colors,
                     node_size=800, font_size=7, arrows=True,
                     arrowsize=15, edge_color='#555555', width=1.5)
    ax.set_title('Bayesian Network Graph')
    ax.axis('off')

    # 2-6 – Marginal bar charts for each variable
    plot_vars = ['Direction', 'RSI_cat', 'Session', 'Trend_cat', 'Volatility_cat']
    labels_map = {
        'Direction': ['Down', 'Up'],
        'RSI_cat': ['Oversold', 'Neutral', 'Overbought'],
        'Session': ['Asian', 'European', 'American'],
        'Trend_cat': ['Below SMA', 'Above SMA'],
        'Volatility_cat': ['Low ATR', 'High ATR'],
    }
    colors_map = {
        'Direction': ['#e74c3c', '#2ecc71'],
        'RSI_cat': ['#3498db', '#f39c12', '#e74c3c'],
        'Session': ['#9b59b6', '#3498db', '#f39c12'],
        'Trend_cat': ['#e74c3c', '#2ecc71'],
        'Volatility_cat': ['#2ecc71', '#e74c3c'],
    }
    flat_axes = axes.flatten()[1:]
    for ax, var in zip(flat_axes, plot_vars):
        counts = df_disc[var].value_counts().sort_index()
        probs = counts / counts.sum()
        lbls = labels_map.get(var, [str(i) for i in counts.index])
        cols = colors_map.get(var, None)
        bars = ax.bar(range(len(probs)), probs.values,
                      color=cols[:len(probs)] if cols else None,
                      edgecolor='white', alpha=0.85)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(lbls, fontsize=8)
        ax.set_ylabel('Marginal Probability')
        ax.set_title(f'Marginal: {var}')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, probs.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_bayesian_network.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def example_inference(infer):
    """Run a few example queries to demonstrate VE inference."""
    print("\n  Example Inference Queries:")

    # Q1: P(Direction | Session=European, RSI=Oversold)
    q1 = infer.query(['Direction'],
                     evidence={'Session': 1, 'RSI_cat': 0},
                     show_progress=False)
    print(f"  P(Direction | Session=European, RSI=Oversold): {np.round(q1.values, 3)}")

    # Q2: P(Direction | Trend=Above SMA, Volume=High)
    q2 = infer.query(['Direction'],
                     evidence={'Trend_cat': 1, 'Volume_cat': 2},
                     show_progress=False)
    print(f"  P(Direction | Trend=AboveSMA, Volume=High): {np.round(q2.values, 3)}")

    # Q3: P(RSI_cat | Session=American, Volatility=High)
    q3 = infer.query(['RSI_cat'],
                     evidence={'Session': 2, 'Volatility_cat': 1},
                     show_progress=False)
    print(f"  P(RSI | Session=American, Volatility=High): {np.round(q3.values, 3)}")


def main():
    print("\n[Task 1] Bayesian Network")
    df_raw = load_data(n_rows=5000)
    df = add_technical_indicators(df_raw)
    df_disc = discretize_for_bn(df)
    print(f"  Discrete dataset shape: {df_disc.shape}")
    print(f"  Variables: {list(df_disc.columns)}")

    model, infer = build_and_fit_bn(df_disc)
    print_cpds(model)
    acc = evaluate_bn(model, infer, df_disc, n_eval=300)
    example_inference(infer)
    plot_bn(model, df_disc)
    return model, infer, {'Accuracy': acc}


if __name__ == '__main__':
    main()
