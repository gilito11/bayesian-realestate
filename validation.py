"""
Out-of-Sample Predictive Validation
====================================

Evaluates the hierarchical model's ability to generalize by splitting
data into train/test sets and measuring predictive accuracy on unseen
listings. Reports both point estimates (RMSE, MAE) and calibration
of uncertainty (credible interval coverage).

This answers the question: "Does the model predict well, or does it
just fit the training data?" — critical for any applied Bayesian model.
"""

import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> tuple:
    """Stratified split preserving portal proportions."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for portal in df["portal"].unique():
        portal_mask = df["portal"] == portal
        portal_indices = df.index[portal_mask].tolist()
        n_test = max(1, int(len(portal_indices) * test_frac))
        selected = rng.choice(portal_indices, size=n_test, replace=False)
        test_idx.extend(selected)

    test_idx = sorted(test_idx)
    train_idx = sorted(set(df.index) - set(test_idx))

    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def _reindex_portals(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Ensure train and test share the same portal encoding."""
    all_portals = sorted(set(train["portal"].unique()) | set(test["portal"].unique()))
    portal_map = {p: i for i, p in enumerate(all_portals)}
    train = train.copy()
    test = test.copy()
    train["portal_idx"] = train["portal"].map(portal_map)
    test["portal_idx"] = test["portal"].map(portal_map)
    return train, test, all_portals


def _standardize_together(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Standardize test set using train set statistics (no data leakage)."""
    train = train.copy()
    test = test.copy()

    for col in ["size_m2", "bedrooms", "bathrooms"]:
        mean, std = train[col].mean(), train[col].std()
        train[f"{col}_z"] = (train[col] - mean) / (std + 1e-8)
        test[f"{col}_z"] = (test[col] - mean) / (std + 1e-8)

    lp_mean, lp_std = train["log_price"].mean(), train["log_price"].std()
    train["log_price_z"] = (train["log_price"] - lp_mean) / lp_std
    test["log_price_z"] = (test["log_price"] - lp_mean) / lp_std

    return train, test, lp_mean, lp_std


def build_train_model(train: pd.DataFrame, n_portals: int) -> pm.Model:
    """Build hierarchical model on training data."""
    portal_idx = train["portal_idx"].values
    X_size = train["size_m2_z"].values
    X_beds = train["bedrooms_z"].values
    X_baths = train["bathrooms_z"].values
    y = train["log_price_z"].values

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5)
        mu_beta_size = pm.Normal("mu_beta_size", mu=0, sigma=5)
        sigma_beta_size = pm.HalfNormal("sigma_beta_size", sigma=3)
        mu_beta_beds = pm.Normal("mu_beta_beds", mu=0, sigma=5)
        sigma_beta_beds = pm.HalfNormal("sigma_beta_beds", sigma=3)
        mu_beta_baths = pm.Normal("mu_beta_baths", mu=0, sigma=5)
        sigma_beta_baths = pm.HalfNormal("sigma_beta_baths", sigma=3)

        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_portals)
        beta_size = pm.Normal("beta_size", mu=mu_beta_size, sigma=sigma_beta_size, shape=n_portals)
        beta_beds = pm.Normal("beta_beds", mu=mu_beta_beds, sigma=sigma_beta_beds, shape=n_portals)
        beta_baths = pm.Normal("beta_baths", mu=mu_beta_baths, sigma=sigma_beta_baths, shape=n_portals)
        sigma = pm.HalfNormal("sigma", sigma=5)

        mu = (
            alpha[portal_idx]
            + beta_size[portal_idx] * X_size
            + beta_beds[portal_idx] * X_beds
            + beta_baths[portal_idx] * X_baths
        )
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

    return model


def predict_test(trace: az.InferenceData, test: pd.DataFrame) -> dict:
    """Generate predictions for test set using posterior samples."""
    post = trace.posterior
    portal_idx = test["portal_idx"].values
    X_size = test["size_m2_z"].values
    X_beds = test["bedrooms_z"].values
    X_baths = test["bathrooms_z"].values

    # Extract posterior samples (chains x draws)
    alpha = post["alpha"].values
    beta_size = post["beta_size"].values
    beta_beds = post["beta_beds"].values
    beta_baths = post["beta_baths"].values
    sigma = post["sigma"].values

    n_chains, n_draws = alpha.shape[:2]
    n_test = len(test)
    predictions = np.zeros((n_chains * n_draws, n_test))

    for c in range(n_chains):
        for d in range(n_draws):
            mu = (
                alpha[c, d, portal_idx]
                + beta_size[c, d, portal_idx] * X_size
                + beta_beds[c, d, portal_idx] * X_beds
                + beta_baths[c, d, portal_idx] * X_baths
            )
            predictions[c * n_draws + d] = mu

    return {
        "pred_mean": predictions.mean(axis=0),
        "pred_std": predictions.std(axis=0),
        "pred_lower": np.percentile(predictions, 3, axis=0),
        "pred_upper": np.percentile(predictions, 97, axis=0),
        "pred_samples": predictions,
    }


def validation_metrics(y_true: np.ndarray, preds: dict) -> dict:
    """Compute RMSE, MAE, and credible interval coverage."""
    residuals = y_true - preds["pred_mean"]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    in_interval = (y_true >= preds["pred_lower"]) & (y_true <= preds["pred_upper"])
    coverage = float(np.mean(in_interval))

    # Interval sharpness (narrower intervals = better, if coverage holds)
    avg_width = float(np.mean(preds["pred_upper"] - preds["pred_lower"]))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "coverage_94pct": round(coverage, 3),
        "avg_interval_width": round(avg_width, 4),
        "n_test": len(y_true),
    }


def run_validation(df: pd.DataFrame, draws=1000, tune=500, chains=2, quick=False):
    """Full validation pipeline: split, train, predict, evaluate."""
    print("  Splitting data 80/20 (stratified by portal)...")
    train_raw, test_raw = train_test_split(df)
    train_raw, test_raw, all_portals = _reindex_portals(train_raw, test_raw)
    train, test, lp_mean, lp_std = _standardize_together(train_raw, test_raw)
    n_portals = len(all_portals)

    print(f"    Train: {len(train)} listings")
    print(f"    Test:  {len(test)} listings")
    print(f"    Portals: {', '.join(all_portals)}")

    print("\n  Training hierarchical model on train set...")
    model = build_train_model(train, n_portals)

    d = 500 if quick else draws
    t = 300 if quick else tune

    t0 = time.perf_counter()
    try:
        import nutpie
        compiled = nutpie.compile_pymc_model(model)
        trace = nutpie.sample(compiled, draws=d, tune=t, chains=chains, seed=42)
    except ImportError:
        with model:
            trace = pm.sample(
                draws=d, tune=t, chains=chains, cores=1,
                random_seed=42, return_inferencedata=True,
            )
    elapsed = time.perf_counter() - t0
    print(f"    Trained in {elapsed:.1f}s")

    print("\n  Predicting on test set...")
    preds = predict_test(trace, test)

    y_test = test["log_price_z"].values
    metrics = validation_metrics(y_test, preds)

    print(f"\n  Out-of-sample metrics (standardized scale):")
    print(f"    RMSE:                {metrics['rmse']:.4f}")
    print(f"    MAE:                 {metrics['mae']:.4f}")
    print(f"    94% CI coverage:     {metrics['coverage_94pct']:.1%}")
    print(f"    Avg interval width:  {metrics['avg_interval_width']:.4f}")

    # Convert back to original EUR scale for interpretability
    pred_eur = np.exp(preds["pred_mean"] * lp_std + lp_mean)
    actual_eur = np.exp(y_test * lp_std + lp_mean)
    mae_eur = float(np.mean(np.abs(actual_eur - pred_eur)))
    mape = float(np.mean(np.abs(actual_eur - pred_eur) / actual_eur) * 100)
    print(f"\n  Original EUR scale:")
    print(f"    MAE:   {mae_eur:,.0f} EUR")
    print(f"    MAPE:  {mape:.1f}%")

    return {
        "train": train, "test": test,
        "trace": trace, "preds": preds,
        "metrics": metrics,
        "lp_mean": lp_mean, "lp_std": lp_std,
        "mae_eur": mae_eur, "mape": mape,
    }


def plot_predictions(val_result: dict, save_path: str = None):
    """Plot predicted vs actual with uncertainty bands."""
    test = val_result["test"]
    preds = val_result["preds"]
    lp_mean = val_result["lp_mean"]
    lp_std = val_result["lp_std"]

    # Convert to EUR
    y_actual = np.exp(test["log_price_z"].values * lp_std + lp_mean)
    y_pred = np.exp(preds["pred_mean"] * lp_std + lp_mean)
    y_lower = np.exp(preds["pred_lower"] * lp_std + lp_mean)
    y_upper = np.exp(preds["pred_upper"] * lp_std + lp_mean)

    sort_idx = np.argsort(y_actual)
    y_actual = y_actual[sort_idx]
    y_pred = y_pred[sort_idx]
    y_lower = y_lower[sort_idx]
    y_upper = y_upper[sort_idx]
    portals = test["portal"].values[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Predicted vs actual scatter
    portal_colors = {
        "habitaclia": "#2196F3", "fotocasa": "#FF9800",
        "milanuncios": "#4CAF50", "idealista": "#E91E63",
    }
    for portal in test["portal"].unique():
        mask = portals == portal
        axes[0].scatter(y_actual[mask], y_pred[mask], s=20, alpha=0.6,
                        color=portal_colors.get(portal, "#666"), label=portal)

    lims = [min(y_actual.min(), y_pred.min()) * 0.9,
            max(y_actual.max(), y_pred.max()) * 1.1]
    axes[0].plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Price (EUR)")
    axes[0].set_ylabel("Predicted Price (EUR)")
    axes[0].set_title("Predicted vs Actual (test set)")
    axes[0].legend(fontsize=8)

    # 2. Predictions with credible intervals
    x = np.arange(len(y_actual))
    axes[1].fill_between(x, y_lower, y_upper, alpha=0.3, color="#2196F3",
                         label="94% credible interval")
    axes[1].plot(x, y_pred, color="#2196F3", linewidth=1, label="Prediction")
    axes[1].scatter(x, y_actual, color="#E91E63", s=10, alpha=0.6, zorder=5,
                    label="Actual")
    axes[1].set_xlabel("Test listings (sorted by price)")
    axes[1].set_ylabel("Price (EUR)")
    axes[1].set_title("Predictions with Uncertainty Bands")
    axes[1].legend(fontsize=8)

    metrics = val_result["metrics"]
    fig.suptitle(
        f"Out-of-Sample Validation — MAE: {val_result['mae_eur']:,.0f} EUR, "
        f"MAPE: {val_result['mape']:.1f}%, "
        f"Coverage: {metrics['coverage_94pct']:.0%}",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
