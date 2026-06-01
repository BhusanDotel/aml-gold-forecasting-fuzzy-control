"""
Gaussian Process Regression and Classification on XAUUSD M15.

Regression:  Predict next-bar log-return from 5 technical indicators.
Classification: Binary direction prediction (up/down) via threshold on GP output.

Inputs (5): RSI, SMA_Ratio, HL_Range, Volume_Norm, ATR
Output (regression): next-bar log-return
Output (classification): 1 = price up, 0 = price down
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_data, add_technical_indicators, GP_FEATURES, GP_TARGET

ASSETS = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def prepare_data(n_rows=3000, test_size=0.2, random_state=42):
    df_raw = load_data(n_rows=n_rows)
    df = add_technical_indicators(df_raw)

    X = df[GP_FEATURES].values
    y_reg = df[GP_TARGET].values
    y_cls = (y_reg > 0).astype(int)
    close = df['Close'].values          # current-bar close price (USD/oz)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, cl_tr, cl_te = train_test_split(
        X_scaled, y_reg, y_cls, close,
        test_size=test_size, random_state=random_state, shuffle=False
    )
    return X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, scaler, cl_te


# ── Gaussian Process Regression ───────────────────────────────────────────────

def run_gp_regression(X_tr, X_te, yr_tr, yr_te, cl_te=None, n_samples=300):
    """Fit GPR on a small subset for computational tractability."""
    idx = np.random.RandomState(0).choice(len(X_tr), min(n_samples, len(X_tr)), replace=False)
    Xr, yr = X_tr[idx], yr_tr[idx]

    kernel = C(1.0) * Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True, random_state=42)
    gpr.fit(Xr, yr)

    # Evaluate on test set (first 200 points)
    Xte = X_te[:200]
    yte = yr_te[:200]
    y_pred, y_std = gpr.predict(Xte, return_std=True)

    mse = mean_squared_error(yte, y_pred)
    mae = mean_absolute_error(yte, y_pred)
    r2 = r2_score(yte, y_pred)

    print("=" * 50)
    print("Gaussian Process Regression Results")
    print("=" * 50)
    print(f"  Kernel (optimised): {gpr.kernel_}")
    print(f"  MSE  : {mse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  Log-likelihood: {gpr.log_marginal_likelihood_value_:.3f}")

    _plot_gpr(yte, y_pred, y_std, mse, r2)

    # Price-level prediction (the part that shows actual USD forecasts)
    if cl_te is not None:
        close_slice = cl_te[:200]
        price_actual = close_slice * np.exp(yte)          # actual next-bar close
        price_pred   = close_slice * np.exp(y_pred)       # GP predicted next-bar close
        price_lo     = close_slice * np.exp(y_pred - 2 * y_std)
        price_hi     = close_slice * np.exp(y_pred + 2 * y_std)
        price_mae    = mean_absolute_error(price_actual, price_pred)
        price_mape   = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100
        print(f"  Price-level MAE : ${price_mae:.2f} / oz")
        print(f"  Price-level MAPE: {price_mape:.4f}%")
        _plot_price_prediction(close_slice, price_actual, price_pred,
                               price_lo, price_hi, price_mae, price_mape)

    return gpr, y_pred, y_std, {'MSE': mse, 'MAE': mae, 'R2': r2}


def _plot_gpr(y_true, y_pred, y_std, mse, r2):
    n = len(y_true)
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Gaussian Process Regression — XAUUSD M15\n'
        'Predicting Next-Bar Log-Return from Technical Indicators',
        fontsize=13, fontweight='bold'
    )

    # 1 Prediction vs actual with uncertainty
    ax = axes[0, 0]
    ax.plot(idx, y_true, 'k-', lw=0.8, label='Actual', alpha=0.7)
    ax.plot(idx, y_pred, 'b-', lw=1.0, label='GP Mean', alpha=0.85)
    ax.fill_between(idx, y_pred - 2 * y_std, y_pred + 2 * y_std,
                    alpha=0.25, color='blue', label='±2σ (95% CI)')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Prediction vs Actual  (R²={r2:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2 Scatter
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color='steelblue', edgecolors='none')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Log Return')
    ax.set_ylabel('Predicted Log Return')
    ax.set_title(f'Predicted vs Actual (MSE={mse:.6f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3 Predictive uncertainty distribution
    ax = axes[1, 0]
    ax.hist(y_std, bins=30, color='orange', edgecolor='white', alpha=0.8)
    ax.axvline(y_std.mean(), color='red', linestyle='--', label=f'Mean σ={y_std.mean():.4f}')
    ax.set_xlabel('Predictive Std (σ)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictive Uncertainty')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4 Residuals
    ax = axes[1, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.35, s=12, color='green', edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel('Predicted Log Return')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_gp_regression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _plot_price_prediction(close, price_actual, price_pred, price_lo, price_hi,
                           price_mae, price_mape):
    """Plot GP-predicted XAUUSD price (USD/oz) vs actual price with uncertainty."""
    n   = len(price_actual)
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'GP-Predicted XAUUSD Gold Price (USD/oz)\n'
        f'MAE = ${price_mae:.2f}/oz  |  MAPE = {price_mape:.4f}%',
        fontsize=13, fontweight='bold'
    )

    # 1 Actual price vs GP predicted price with 95% CI band
    ax = axes[0, 0]
    ax.plot(idx, price_actual, 'k-', lw=1.2, label='Actual Close (USD/oz)', alpha=0.85)
    ax.plot(idx, price_pred,   'r-', lw=1.5, label='GP Predicted Close', alpha=0.9)
    ax.fill_between(idx, price_lo, price_hi,
                    alpha=0.2, color='red', label='±2σ Confidence Band')
    ax.set_xlabel('Test Bar Index')
    ax.set_ylabel('Gold Price (USD/oz)')
    ax.set_title('Actual vs GP-Predicted XAUUSD Price')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # annotate price range
    ax.text(0.02, 0.97, f'Price range: ${price_actual.min():.0f}–${price_actual.max():.0f}',
            transform=ax.transAxes, va='top', fontsize=8, color='grey')

    # 2 Prediction error in USD
    ax = axes[0, 1]
    error_usd = price_pred - price_actual
    ax.bar(idx, error_usd, color=np.where(error_usd >= 0, '#2ecc71', '#e74c3c'),
           width=1.0, alpha=0.7, label='Pred − Actual (USD)')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(price_mae,  color='blue', lw=1.2, linestyle='--', label=f'+MAE ${price_mae:.2f}')
    ax.axhline(-price_mae, color='blue', lw=1.2, linestyle='--', label=f'−MAE ${price_mae:.2f}')
    ax.set_xlabel('Test Bar Index')
    ax.set_ylabel('Price Error (USD/oz)')
    ax.set_title('Prediction Error in USD')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3 Scatter: predicted vs actual price
    ax = axes[1, 0]
    ax.scatter(price_actual, price_pred, alpha=0.35, s=12,
               c=np.abs(error_usd), cmap='RdYlGn_r', edgecolors='none')
    lo, hi = price_actual.min() - 2, price_actual.max() + 2
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect forecast')
    ax.set_xlabel('Actual Price (USD/oz)')
    ax.set_ylabel('Predicted Price (USD/oz)')
    ax.set_title(f'Predicted vs Actual Price (MAE=${price_mae:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4 Confidence band width over time (shows where GP is uncertain)
    ax = axes[1, 1]
    band_width = price_hi - price_lo
    ax.plot(idx, band_width, color='darkorange', lw=1.5, label='95% CI width (USD)')
    ax.fill_between(idx, 0, band_width, alpha=0.2, color='orange')
    ax.axhline(band_width.mean(), color='red', linestyle='--', lw=1.2,
               label=f'Mean width ${band_width.mean():.2f}')
    ax.set_xlabel('Test Bar Index')
    ax.set_ylabel('95% CI Width (USD/oz)')
    ax.set_title('GP Predictive Uncertainty Band Width')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_gp_price_prediction.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Gaussian Process Classification ──────────────────────────────────────────

def run_gp_classification(X_tr, X_te, yc_tr, yc_te, n_samples=300):
    """GPC on direction labels derived by thresholding GP regression output."""
    idx = np.random.RandomState(1).choice(len(X_tr), min(n_samples, len(X_tr)), replace=False)
    Xc, yc = X_tr[idx], yc_tr[idx]

    kernel = C(1.0) * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=3,
                                    random_state=42, max_iter_predict=100)
    gpc.fit(Xc, yc)

    Xte = X_te[:200]
    yte = yc_te[:200]
    y_pred_cls = gpc.predict(Xte)
    y_prob = gpc.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, y_pred_cls)
    print("\n" + "=" * 50)
    print("Gaussian Process Classification Results")
    print("=" * 50)
    print(f"  Kernel (optimised): {gpc.kernel_}")
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(yte, y_pred_cls, target_names=['Down (0)', 'Up (1)']))

    _plot_gpc(yte, y_pred_cls, y_prob, acc)
    return gpc, {'Accuracy': acc}


def _plot_gpc(y_true, y_pred, y_prob, acc):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Gaussian Process Classification — XAUUSD M15\n'
        'Predicting Next-Bar Price Direction (Up / Down)',
        fontsize=13, fontweight='bold'
    )

    # 1 Predicted probabilities
    ax = axes[0]
    ax.plot(y_prob, 'o', ms=3, color='steelblue', alpha=0.6, label='P(Up)')
    ax.axhline(0.5, color='red', linestyle='--', lw=1.2, label='Decision boundary')
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            ax.axvline(i, color='orange', alpha=0.3, lw=0.8)
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Predicted P(Up)')
    ax.set_title(f'Predictive Probability (Acc={acc:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2 Confusion matrix
    ax = axes[1]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Down', 'Pred Up'])
    ax.set_yticklabels(['Actual Down', 'Actual Up'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)

    # 3 Probability histogram per class
    ax = axes[2]
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, color='red', label='Actual Down')
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, color='green', label='Actual Up')
    ax.axvline(0.5, color='black', linestyle='--', lw=1.2)
    ax.set_xlabel('P(Up)')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution by Class')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ASSETS, 'task1_gp_classification.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[Task 1] Gaussian Process Regression & Classification")
    print("Loading and preparing data...")
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, scaler, cl_te = prepare_data(n_rows=3000)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"  Features: RSI, SMA_Ratio, HL_Range, Volume_Norm, ATR")
    print(f"  Target (reg): next-bar log-return → converted to USD/oz price")
    print(f"  Target (cls): direction class (1=up / 0=down)")

    gpr, y_pred_reg, y_std_reg, reg_metrics = run_gp_regression(
        X_tr, X_te, yr_tr, yr_te, cl_te=cl_te
    )
    gpc, cls_metrics = run_gp_classification(X_tr, X_te, yc_tr, yc_te)

    return gpr, gpc, reg_metrics, cls_metrics


if __name__ == '__main__':
    main()
