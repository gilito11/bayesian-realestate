"""
Prior Sensitivity Analysis
==========================

Tests whether posterior conclusions are robust to the choice of priors.
Runs the hierarchical model under three prior regimes and compares
the resulting posteriors — if they converge to similar values, the
data is informative enough to overwhelm the prior specification.

Prior sets:
  1. Weakly informative (default): Normal(0, 10), HalfNormal(5)
  2. Diffuse / vague:              Normal(0, 100), HalfNormal(50)
  3. Informative / tight:          Normal(0, 1), HalfNormal(0.5)

A robust model should show similar posteriors across all three,
demonstrating that conclusions are data-driven, not prior-driven.
"""

import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

PRIOR_SETS = {
    "weakly_informative": {
        "mu_sigma": 10, "alpha_sigma": 5,
        "beta_sigma": 5, "beta_group_sigma": 3,
        "obs_sigma": 5,
        "label": "Weakly Informative (default)",
    },
    "diffuse": {
        "mu_sigma": 100, "alpha_sigma": 50,
        "beta_sigma": 50, "beta_group_sigma": 30,
        "obs_sigma": 50,
        "label": "Diffuse / Vague",
    },
    "informative": {
        "mu_sigma": 1, "alpha_sigma": 0.5,
        "beta_sigma": 1, "beta_group_sigma": 0.5,
        "obs_sigma": 1,
        "label": "Informative / Tight",
    },
}


def build_with_priors(df: pd.DataFrame, prior_set: dict) -> pm.Model:
    """Build the hierarchical model with a specific prior configuration."""
    portal_idx = df["portal_idx"].values
    n_portals = df["portal_idx"].nunique()
    X_size = df["size_m2_z"].values
    X_beds = df["bedrooms_z"].values
    X_baths = df["bathrooms_z"].values
    y = df["log_price_z"].values

    ps = prior_set

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=ps["mu_sigma"])
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=ps["alpha_sigma"])

        mu_beta_size = pm.Normal("mu_beta_size", mu=0, sigma=ps["beta_sigma"])
        sigma_beta_size = pm.HalfNormal("sigma_beta_size", sigma=ps["beta_group_sigma"])

        mu_beta_beds = pm.Normal("mu_beta_beds", mu=0, sigma=ps["beta_sigma"])
        sigma_beta_beds = pm.HalfNormal("sigma_beta_beds", sigma=ps["beta_group_sigma"])

        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_portals)
        beta_size = pm.Normal(
            "beta_size", mu=mu_beta_size, sigma=sigma_beta_size, shape=n_portals
        )
        beta_beds = pm.Normal(
            "beta_beds", mu=mu_beta_beds, sigma=sigma_beta_beds, shape=n_portals
        )

        sigma = pm.HalfNormal("sigma", sigma=ps["obs_sigma"])

        mu = (
            alpha[portal_idx]
            + beta_size[portal_idx] * X_size
            + beta_beds[portal_idx] * X_beds
        )

        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

    return model


def run_sensitivity(df: pd.DataFrame, draws=1000, tune=500, chains=2) -> dict:
    """Run the hierarchical model under all three prior regimes.

    Returns a dict {prior_name: {"trace": InferenceData, "time": float, "priors": dict}}.
    """
    results = {}

    for name, prior_set in PRIOR_SETS.items():
        print(f"\n  [{prior_set['label']}]")
        model = build_with_priors(df, prior_set)

        t0 = time.perf_counter()
        try:
            import nutpie
            compiled = nutpie.compile_pymc_model(model)
            trace = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=42)
        except ImportError:
            with model:
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains, cores=1,
                    random_seed=42, return_inferencedata=True, progressbar=True,
                )
        elapsed = time.perf_counter() - t0
        print(f"    Sampled in {elapsed:.1f}s")

        results[name] = {
            "trace": trace,
            "time": elapsed,
            "priors": prior_set,
        }

    return results


def sensitivity_summary(results: dict) -> pd.DataFrame:
    """Compare posterior means across prior specifications."""
    key_vars = ["mu_alpha", "mu_beta_size", "sigma"]
    rows = []

    for prior_name, res in results.items():
        post = res["trace"].posterior
        row = {"prior_set": PRIOR_SETS[prior_name]["label"]}
        for var in key_vars:
            row[f"{var}_mean"] = round(float(post[var].mean()), 4)
            hdi = az.hdi(post[var].values.flatten(), hdi_prob=0.94)
            row[f"{var}_hdi_width"] = round(float(hdi[1] - hdi[0]), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def sensitivity_divergence(results: dict) -> dict:
    """Measure how much posteriors diverge across prior sets.

    Returns max absolute difference in posterior means for each key variable.
    Small values (<0.1 for standardized data) indicate robustness.
    """
    key_vars = ["mu_alpha", "mu_beta_size", "sigma"]
    means = {var: [] for var in key_vars}

    for res in results.values():
        post = res["trace"].posterior
        for var in key_vars:
            means[var].append(float(post[var].mean()))

    return {
        var: round(max(vals) - min(vals), 4)
        for var, vals in means.items()
    }


def plot_sensitivity(results: dict, save_path: str = None):
    """Overlay posterior densities from different prior specifications."""
    import matplotlib.pyplot as plt

    key_vars = ["mu_alpha", "mu_beta_size", "sigma"]
    colors = {"weakly_informative": "#2196F3", "diffuse": "#FF9800", "informative": "#4CAF50"}

    fig, axes = plt.subplots(1, len(key_vars), figsize=(5 * len(key_vars), 4))

    for ax, var in zip(axes, key_vars):
        for prior_name, res in results.items():
            samples = res["trace"].posterior[var].values.flatten()
            label = PRIOR_SETS[prior_name]["label"]
            ax.hist(samples, bins=60, density=True, alpha=0.4,
                    color=colors[prior_name], label=label)
        ax.set_title(var)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Prior Sensitivity: Posterior Distributions Under Different Priors",
                 fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
