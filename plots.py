"""
Visualization module for Bayesian Real Estate Intelligence.

Generates publication-quality plots using ArviZ and matplotlib.
All plot functions return (fig, ax) for composability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import arviz as az

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PORTAL_COLORS = {
    "habitaclia": "#2196F3",
    "fotocasa": "#FF9800",
    "milanuncios": "#4CAF50",
    "idealista": "#E91E63",
}


def plot_shrinkage(shrinkage_df: pd.DataFrame, save_path: str = None):
    """Visualize partial pooling: portal estimates shrunk toward group mean."""
    fig, ax = plt.subplots(figsize=(10, 5))
    group_mean = shrinkage_df["group_mu_alpha"].iloc[0]

    for i, row in shrinkage_df.iterrows():
        color = PORTAL_COLORS.get(row["portal"], "#666")
        ax.errorbar(
            row["alpha_mean"], i,
            xerr=[[row["alpha_mean"] - row["alpha_hdi_low"]],
                  [row["alpha_hdi_high"] - row["alpha_mean"]]],
            fmt="o", color=color, capsize=5, markersize=8, linewidth=2,
            label=row["portal"],
        )

    ax.axvline(group_mean, color="red", linestyle="--", alpha=0.7, label="Group mean")
    ax.set_yticks(range(len(shrinkage_df)))
    ax.set_yticklabels(shrinkage_df["portal"])
    ax.set_xlabel("Intercept (log-price, standardized)")
    ax.set_title("Hierarchical Shrinkage: Portal Intercepts toward Group Mean")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_posterior_comparison(trace, var_names, title="Posterior Distributions",
                              save_path=None):
    """Side-by-side posterior density plots."""
    axes = az.plot_posterior(trace, var_names=var_names, figsize=(14, 4))
    fig = axes.flatten()[0].get_figure()
    fig.suptitle(title, fontsize=14, y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spatial_surface(grid_data: dict, df: pd.DataFrame, save_path: str = None):
    """Plot GP-predicted price surface with uncertainty."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean surface
    im1 = axes[0].contourf(
        grid_data["lon_grid"], grid_data["lat_grid"],
        grid_data["f_mean"], levels=20, cmap="RdYlGn_r",
    )
    axes[0].scatter(df["lon"], df["lat"], c="black", s=5, alpha=0.3, zorder=5)
    axes[0].set_title("Predicted Price Surface (mean)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="Log-price (standardized)")

    # Uncertainty surface
    im2 = axes[1].contourf(
        grid_data["lon_grid"], grid_data["lat_grid"],
        grid_data["f_std"], levels=20, cmap="Oranges",
    )
    axes[1].scatter(df["lon"], df["lat"], c="black", s=5, alpha=0.3, zorder=5)
    axes[1].set_title("Prediction Uncertainty (std)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="Std deviation")

    fig.suptitle("Gaussian Process Spatial Price Model", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_anomaly_scores(scores_df: pd.DataFrame, save_path: str = None):
    """Visualize anomaly detection results."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Anomaly score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores_df["p_anomaly"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax1.axvline(0.5, color="red", linestyle="--", alpha=0.8, label="Decision boundary")
    ax1.set_xlabel("P(anomaly)")
    ax1.set_ylabel("Count")
    ax1.set_title("Anomaly Score Distribution")
    ax1.legend()

    # 2. Price vs size colored by anomaly score
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(
        scores_df["size_m2"], scores_df["price"],
        c=scores_df["p_anomaly"], cmap="RdYlGn_r", s=20, alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax2, label="P(anomaly)")
    ax2.set_xlabel("Size (m²)")
    ax2.set_ylabel("Price (EUR)")
    ax2.set_title("Listings by Anomaly Probability")

    # 3. Top anomalies by portal
    ax3 = fig.add_subplot(gs[1, 0])
    flagged = scores_df[scores_df["is_flagged"]]
    if len(flagged) > 0:
        portal_counts = flagged["portal"].value_counts()
        colors = [PORTAL_COLORS.get(p, "#666") for p in portal_counts.index]
        ax3.barh(portal_counts.index, portal_counts.values, color=colors)
        ax3.set_xlabel("Flagged listings")
        ax3.set_title("Anomalies by Portal")
    else:
        ax3.text(0.5, 0.5, "No anomalies detected", ha="center", va="center")

    # 4. Residual distribution with mixture
    ax4 = fig.add_subplot(gs[1, 1])
    normal_mask = ~scores_df["is_flagged"]
    ax4.hist(scores_df.loc[normal_mask, "residual"], bins=40, alpha=0.6,
             color="#4CAF50", label="Normal", density=True)
    if len(flagged) > 0:
        ax4.hist(scores_df.loc[~normal_mask, "residual"], bins=20, alpha=0.6,
                 color="#E91E63", label="Anomaly", density=True)
    ax4.set_xlabel("Price residual")
    ax4.set_title("Residual Distribution by Component")
    ax4.legend()

    fig.suptitle("Bayesian Anomaly Detection Results", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison_summary(models: dict, save_path: str = None):
    """Visual summary of model comparison and tractability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(models.keys())
    nuts_times = [getattr(m, "nuts_time", 0) for m in models.values()]
    advi_times = [getattr(m, "advi_time", 0) for m in models.values()]

    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x - width/2, nuts_times, width, label="NUTS (MCMC)", color="#2196F3")
    if any(t > 0 for t in advi_times):
        axes[0].bar(x + width/2, advi_times, width, label="ADVI (VI)", color="#FF9800")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Inference Time: MCMC vs VI")
    axes[0].legend()

    # ESS/s comparison
    ess_rates = []
    for m in models.values():
        if hasattr(m, "trace") and m.trace is not None and hasattr(m, "nuts_time"):
            summary = az.summary(m.trace, kind="diagnostics")
            ess_rates.append(float(summary["ess_bulk"].median() / m.nuts_time))
        else:
            ess_rates.append(0)

    colors = ["#4CAF50" if e > 50 else "#FF9800" if e > 20 else "#E91E63" for e in ess_rates]
    axes[1].bar(names, ess_rates, color=colors)
    axes[1].set_ylabel("ESS / second")
    axes[1].set_title("Sampling Efficiency")
    axes[1].axhline(50, color="green", linestyle="--", alpha=0.5, label="Good (>50)")
    axes[1].axhline(20, color="orange", linestyle="--", alpha=0.5, label="Marginal (>20)")
    axes[1].legend()

    fig.suptitle("Tractability Analysis", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
