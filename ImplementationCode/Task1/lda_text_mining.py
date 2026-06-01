"""
Latent Dirichlet Allocation for financial text mining.

Corpus: 300 synthetic financial news documents covering gold/FX market topics.
Topics discovered (k=5):
  T1 — Geopolitical risk & safe-haven demand
  T2 — Federal Reserve / monetary policy
  T3 — Inflation hedge & macroeconomics
  T4 — Technical analysis & price levels
  T5 — Supply, mining & physical demand

Implements full LDA pipeline:
  preprocessing → TF-IDF / BoW → LDA (sklearn) → evaluation (perplexity, coherence proxy)
  → visualisations (topic-word heatmap, doc-topic distribution, t-SNE)
"""

import os, re, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

# ── Synthetic financial news corpus ──────────────────────────────────────────
# 5 thematic groups × 60 documents each = 300 documents

_TEMPLATES = {
    0: [   # Geopolitical risk & safe-haven
        "Gold prices surged as geopolitical tensions escalated in the Middle East, driving investors to safe-haven assets.",
        "Rising geopolitical uncertainty pushed gold demand higher as risk-off sentiment swept global markets.",
        "Investors flocked to gold amid fears of military conflict, lifting spot prices above key resistance levels.",
        "Safe-haven demand for gold intensified following escalating diplomatic tensions between major powers.",
        "Gold rallied sharply as geopolitical risks dominated market sentiment, overshadowing equity gains.",
        "Conflict in eastern Europe renewed safe-haven demand, sending gold to multi-month highs.",
        "Military tensions between regional powers boosted gold prices as investors sought portfolio protection.",
        "Geopolitical anxiety drove bullion buying despite a stronger US dollar, reflecting extreme risk aversion.",
        "Gold futures climbed as uncertainty around upcoming elections increased hedging demand.",
        "Trade war fears rekindled investor appetite for gold as a reliable store of value during crises.",
        "Gold attracted capital flows as global risk appetite deteriorated amid trade sanctions.",
        "Central bank gold purchases accelerated in response to heightened geopolitical uncertainty.",
        "War risk premium embedded in gold prices reflects ongoing uncertainty in commodity markets.",
        "Sanctions imposed on major economies strengthened the case for gold as a reserve asset.",
        "Investors diversified into gold after equity markets fell sharply on geopolitical news.",
        "Gold demand spiked as international relations deteriorated, pushing prices near record highs.",
        "The threat of escalation in regional conflicts underpinned bullion's safe-haven appeal.",
        "Analysts noted that gold's role as a crisis hedge remains intact despite rising real yields.",
        "Gold regained its status as the premier safe-haven after financial market turmoil intensified.",
        "Political instability in emerging markets fuelled capital flight into gold and US Treasuries.",
    ],
    1: [   # Federal Reserve / monetary policy
        "Federal Reserve signalled fewer rate cuts ahead, triggering a gold sell-off as real yields rose.",
        "Gold weakened after the FOMC minutes revealed a more hawkish stance on inflation control.",
        "Rate hike expectations weighed on gold prices, as higher opportunity costs reduced bullion appeal.",
        "Dovish comments from the Fed Chair sparked a gold rally, with traders pricing in earlier rate cuts.",
        "Gold investors closely watched the Federal Reserve balance sheet for signs of quantitative tightening.",
        "Fed officials debated the pace of monetary tightening, creating volatility in gold markets.",
        "A pause in interest rate hikes boosted gold as the dollar softened and real rates declined.",
        "Gold responded positively to hints from central bankers about peaking interest rates.",
        "Quantitative easing expectations revived gold's appeal as a hedge against currency debasement.",
        "Minutes from the last FOMC meeting revealed policymakers concerned about economic slowdown, supporting gold.",
        "The Fed's forward guidance on rates continued to dominate short-term gold price movements.",
        "Expectations of a terminal rate cut drove gold to its highest level in over a year.",
        "Traders repriced gold futures after the Federal Reserve held rates steady at its latest meeting.",
        "Gold prices fell when the Fed announced a larger-than-expected rate increase of 75 basis points.",
        "Central bank communication about the pace of policy normalisation remained the primary gold driver.",
        "Market participants bid up gold prices after the Fed signalled it would slow the pace of tightening.",
        "Tightening financial conditions from consecutive rate hikes pressured speculative gold positions.",
        "Gold fell to multi-month lows as strong US employment data reinforced the case for rate hikes.",
        "The Federal Open Market Committee's decision surprised markets, sending gold sharply higher.",
        "ECB and Fed divergence in monetary policy created cross-currency flows that supported gold.",
    ],
    2: [   # Inflation hedge & macroeconomics
        "Surging CPI data bolstered gold's appeal as an inflation hedge, sending prices to fresh highs.",
        "Persistent inflation above the central bank target renewed institutional interest in gold holdings.",
        "Gold outperformed equities during high-inflation regimes, confirming its role as a real asset hedge.",
        "Negative real interest rates created a favourable environment for gold as a store of wealth.",
        "Consumer price pressures remained elevated, sustaining demand for inflation-hedging instruments like gold.",
        "Gold demand from exchange-traded funds rose sharply as investors sought inflation protection.",
        "Stagflation fears revived interest in gold as a diversifier against stagnant growth and rising prices.",
        "Economists projected inflation would remain above target for another two years, boosting gold forecasts.",
        "Physical gold demand from jewellery and investment surged as real household purchasing power eroded.",
        "Currency depreciation in emerging markets drove retail investors towards gold for wealth preservation.",
        "Gold's inverse correlation with real yields strengthened during the latest inflationary episode.",
        "Inflation breakevens widened, signalling that markets expected prolonged price pressures ahead.",
        "Central banks increased gold reserves as a hedge against fiat currency inflation risks.",
        "Commodity inflation from energy and food prices historically precedes rising gold market activity.",
        "Institutional portfolios raised gold allocations to mitigate the impact of unexpected inflation shocks.",
        "Long-term gold price forecasts were revised upward on persistent above-target inflation expectations.",
        "Pension funds increased exposure to gold as inflation eroded the real value of fixed-income holdings.",
        "Gold prices tracked closely with five-year forward inflation expectations over the past quarter.",
        "Elevated money supply growth raised concerns about currency debasement, supporting gold prices.",
        "Portfolio managers cited gold as an essential hedge in an era of fiscal dominance and debt monetisation.",
    ],
    3: [   # Technical analysis & price levels
        "Gold broke above the critical 2000 resistance level, confirming a bullish breakout on daily charts.",
        "Technical analysts noted a golden cross formation as the 50-day SMA crossed above the 200-day SMA.",
        "The RSI indicator entered overbought territory, suggesting a near-term pullback in gold prices.",
        "Gold found strong support near the 1900 level, with buyers emerging each time prices tested it.",
        "Fibonacci retracement levels highlighted key support zones at 38.2% and 61.8% for gold.",
        "A head-and-shoulders pattern on the weekly gold chart signalled potential trend reversal risk.",
        "Trading volume surged as gold broke through key resistance, confirming momentum in the bullish move.",
        "Bollinger Band contraction in gold charts preceded an explosive price move in recent sessions.",
        "Momentum indicators pointed to continued upside for gold if the weekly close held above 1980.",
        "A descending channel formation in gold suggested consolidation before the next directional move.",
        "Chart patterns showed gold in an ascending triangle, with a measured target near 2100 on breakout.",
        "Moving average convergence divergence (MACD) generated a buy signal as the histogram crossed zero.",
        "Gold's price action confirmed a double-bottom reversal pattern, attracting technical buyers.",
        "Average true range expansion indicated increasing volatility, historically preceding major moves in gold.",
        "Stochastic oscillator divergence from price action provided an early warning of momentum exhaustion.",
        "Key pivot levels for gold were established at 1950 support and 2020 resistance in the near term.",
        "Daily candlestick patterns showed a bullish engulfing at support, suggesting buying interest at lows.",
        "Volume-weighted average price analysis indicated institutional accumulation in gold over recent weeks.",
        "Gold's relative strength against silver provided a confirming signal of safe-haven buying pressure.",
        "Commodity channel index reached extreme levels in gold, historically associated with mean reversion.",
    ],
    4: [   # Supply, mining & physical demand
        "Global gold mine production remained constrained as higher energy costs squeezed mining margins.",
        "Jewellery demand from China and India drove physical gold imports to a quarterly record.",
        "Gold recycling volumes increased as high prices incentivised scrap supply to offset mine shortfalls.",
        "New mine discoveries declined sharply, creating long-term supply concerns in the gold industry.",
        "Central bank gold purchases reached their highest annual total since the Bretton Woods era.",
        "Exploration capex in the gold mining sector fell, raising questions about future supply pipelines.",
        "Seasonal demand from Hindu wedding festivals lifted Indian gold imports in the fourth quarter.",
        "Mining strikes disrupted production at several major South African gold operations.",
        "Environmental regulations tightened operational constraints on new gold mine development approvals.",
        "Technology sector gold demand for semiconductors and electronics continued to grow steadily.",
        "Refinery throughput data showed rising global interest in converting gold bars to investment products.",
        "Physical gold demand from central banks in emerging markets accelerated de-dollarisation trends.",
        "All-in sustaining costs at major gold mines increased, compressing margins at lower price levels.",
        "Gold ETF holdings declined as institutional investors rotated capital towards equities and crypto.",
        "Mining companies reported grade dilution at ageing deposits, reducing average production quality.",
        "Gold royalty companies benefited from rising prices without bearing direct operational cost increases.",
        "Supply disruptions from major producing nations created near-term tightness in gold bullion markets.",
        "Chinese gold consumption grew by double digits, supported by rising middle-class wealth accumulation.",
        "The gold-silver ratio spiked, reflecting disproportionate investment demand relative to industrial uses.",
        "Analyst forecasts for gold mine output over the next decade were revised downward citing reserve depletion.",
    ],
}


def build_corpus(seed=42):
    """Generate 300 synthetic financial news documents (60 per topic)."""
    random.seed(seed)
    np.random.seed(seed)
    docs, labels = [], []
    for topic_id, templates in _TEMPLATES.items():
        for _ in range(60):
            base = random.choice(templates)
            # Light augmentation: shuffle a word or add a filler phrase
            words = base.split()
            if len(words) > 8:
                i = random.randint(1, len(words) - 2)
                words.insert(i, random.choice(['significantly', 'sharply', 'modestly', '']))
            doc = ' '.join(w for w in words if w)
            docs.append(doc)
            labels.append(topic_id)
    return docs, labels


def preprocess(docs):
    """Lowercase, strip punctuation, remove stopwords."""
    cleaned = []
    stop = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'that', 'this',
        'it', 'its', 'their', 'they', 'we', 'he', 'she', 'after', 'before',
        'above', 'below', 'over', 'under', 'about', 'into', 'through', 'during',
        'not', 'no', 'nor', 'so', 'yet', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'than', 'then', 'there', 'these', 'those',
        'what', 'which', 'who', 'how', 'when', 'where', 'while', 'also', 'up',
        'out', 'if', 'near', 'all', 'any', 'can', 'just', 'new',
    }
    for doc in docs:
        doc = doc.lower()
        doc = re.sub(r'[^a-z\s]', ' ', doc)
        tokens = [w for w in doc.split() if w not in stop and len(w) > 2]
        cleaned.append(' '.join(tokens))
    return cleaned


TOPIC_LABELS = [
    'Geopolitical Risk / Safe-Haven',
    'Federal Reserve / Monetary Policy',
    'Inflation Hedge / Macroeconomics',
    'Technical Analysis / Price Levels',
    'Supply, Mining & Physical Demand',
]


def run_lda(docs_clean, n_topics=5, max_features=500):
    vectorizer = CountVectorizer(max_features=max_features, min_df=3,
                                 max_df=0.95, ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(docs_clean)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        learning_method='online',
        learning_offset=50.0,
        random_state=42
    )
    doc_topics = lda.fit_transform(dtm)

    perplexity = lda.perplexity(dtm)
    print("\n" + "=" * 50)
    print("LDA Text Mining Results")
    print("=" * 50)
    print(f"  Vocabulary size  : {len(vectorizer.vocabulary_)}")
    print(f"  Documents        : {dtm.shape[0]}")
    print(f"  Topics (k)       : {n_topics}")
    print(f"  Perplexity       : {perplexity:.2f}")

    feature_names = vectorizer.get_feature_names_out()
    print("\n  Top 10 words per topic:")
    for i, comp in enumerate(lda.components_):
        top10 = [feature_names[j] for j in comp.argsort()[-10:][::-1]]
        print(f"  Topic {i+1} [{TOPIC_LABELS[i]}]:")
        print(f"    {', '.join(top10)}")

    return lda, vectorizer, dtm, doc_topics


def plot_lda(lda, vectorizer, dtm, doc_topics, true_labels):
    feature_names = vectorizer.get_feature_names_out()
    n_topics = lda.n_components

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Latent Dirichlet Allocation — Financial News Text Mining\n'
        'Gold Market Topic Discovery (k=5 Topics, 300 Documents)',
        fontsize=13, fontweight='bold'
    )

    # 1 – Topic-word heatmap (top 15 words per topic)
    ax1 = fig.add_subplot(2, 2, 1)
    top_n = 15
    top_words_per_topic = []
    all_top = set()
    for comp in lda.components_:
        tw = list(comp.argsort()[-top_n:][::-1])
        top_words_per_topic.append(tw)
        all_top.update(tw)
    top_words = sorted(all_top)
    heat = lda.components_[:, top_words]
    heat_norm = heat / heat.sum(axis=1, keepdims=True)
    im = ax1.imshow(heat_norm, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(top_words)))
    ax1.set_xticklabels([feature_names[i] for i in top_words],
                        rotation=90, fontsize=6)
    ax1.set_yticks(range(n_topics))
    ax1.set_yticklabels([f'T{i+1}' for i in range(n_topics)], fontsize=9)
    ax1.set_title('Topic–Word Weight Heatmap')
    plt.colorbar(im, ax=ax1, fraction=0.03)

    # 2 – Document-topic distribution (stacked bar, first 60 docs)
    ax2 = fig.add_subplot(2, 2, 2)
    sample = doc_topics[:60]
    colors = list(mcolors.TABLEAU_COLORS.values())[:n_topics]
    bottom = np.zeros(len(sample))
    for t in range(n_topics):
        ax2.bar(range(len(sample)), sample[:, t], bottom=bottom,
                color=colors[t], label=f'T{t+1}', alpha=0.85)
        bottom += sample[:, t]
    ax2.set_xlabel('Document Index (first 60)')
    ax2.set_ylabel('Topic Proportion')
    ax2.set_title('Document–Topic Distribution (first 60 docs)')
    ax2.legend(fontsize=7, loc='upper right', ncol=2)
    ax2.grid(True, axis='y', alpha=0.3)

    # 3 – t-SNE of document-topic distributions
    ax3 = fig.add_subplot(2, 2, 3)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb = tsne.fit_transform(doc_topics)
    cmap = plt.cm.get_cmap('tab10', n_topics)
    for t in range(n_topics):
        mask = np.array(true_labels) == t
        ax3.scatter(emb[mask, 0], emb[mask, 1], s=20, alpha=0.7,
                    c=[cmap(t)], label=f'T{t+1}: {TOPIC_LABELS[t][:22]}')
    ax3.set_title('t-SNE of Document Topic Distributions')
    ax3.legend(fontsize=6, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('t-SNE dim 1')
    ax3.set_ylabel('t-SNE dim 2')

    # 4 – Top words per topic (horizontal bars)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    y_pos = 1.0
    step = 1.0 / (n_topics * 6 + 2)
    for t in range(n_topics):
        comp = lda.components_[t]
        tw_idx = comp.argsort()[-8:][::-1]
        tw_words = [feature_names[i] for i in tw_idx]
        tw_vals = comp[tw_idx] / comp[tw_idx].sum()
        ax4.text(0, y_pos, f'Topic {t+1}: {TOPIC_LABELS[t]}',
                 fontsize=9, fontweight='bold', color=colors[t],
                 transform=ax4.transAxes)
        y_pos -= step
        for w, v in zip(tw_words, tw_vals):
            ax4.text(0.05, y_pos, f'  {w} ({v:.2f})', fontsize=7,
                     transform=ax4.transAxes)
            y_pos -= step * 0.9
        y_pos -= step * 0.5
    ax4.set_title('Top Words per Topic', pad=10)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_lda_text_mining.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_topic_coherence_proxy(lda, dtm, vectorizer):
    """Plot per-topic top-word probability distribution as coherence proxy."""
    feature_names = vectorizer.get_feature_names_out()
    n_topics = lda.n_components
    colors = list(mcolors.TABLEAU_COLORS.values())[:n_topics]

    fig, axes = plt.subplots(1, n_topics, figsize=(18, 4), sharey=False)
    fig.suptitle('Top-10 Word Weights per LDA Topic', fontsize=12, fontweight='bold')

    for t, ax in enumerate(axes):
        comp = lda.components_[t]
        top10_idx = comp.argsort()[-10:][::-1]
        words = [feature_names[i] for i in top10_idx]
        vals = comp[top10_idx] / comp[top10_idx].sum()
        ax.barh(range(len(words)), vals[::-1], color=colors[t], alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words[::-1], fontsize=7)
        ax.set_title(f'T{t+1}\n{TOPIC_LABELS[t][:20]}', fontsize=8)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_lda_topic_words.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("\n[Task 1] LDA Text Mining")
    docs_raw, true_labels = build_corpus()
    print(f"  Corpus size: {len(docs_raw)} documents, {len(set(true_labels))} ground-truth topics")

    docs_clean = preprocess(docs_raw)
    lda, vectorizer, dtm, doc_topics = run_lda(docs_clean, n_topics=5)
    plot_lda(lda, vectorizer, dtm, doc_topics, true_labels)
    plot_topic_coherence_proxy(lda, dtm, vectorizer)

    # Topic assignment accuracy against ground-truth (best-match permutation)
    pred_topics = doc_topics.argmax(axis=1)
    from scipy.optimize import linear_sum_assignment
    cm = np.zeros((5, 5), dtype=int)
    for p, t in zip(pred_topics, true_labels):
        cm[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / len(true_labels)
    print(f"\n  Topic assignment accuracy (best permutation): {acc:.4f}")
    return lda, {'Topic_Assignment_Accuracy': acc}


if __name__ == '__main__':
    main()
