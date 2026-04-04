#!/usr/bin/env python3
"""
Bayesian Real Estate Intelligence - Full Demo
==============================================

Runs all three probabilistic models on real estate data,
produces diagnostics, tractability analysis, and visualizations.

Usage:
    python demo.py                           # synthetic data
    python demo.py --source neon             # Neon PostgreSQL (requires DATABASE_URL)
    python demo.py --no-spatial              # skip GP model (slow)
    python demo.py --quick                   # fast mode: fewer samples
"""

import argparse
import os
import sys
import time
import warnings

# Use numba backend (no C compiler needed, much faster than Python mode)
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pymc")

from data import generate_synthetic, load_from_neon, preprocess
from models import HierarchicalPricingModel, SpatialGPModel, AnomalyMixtureModel
from diagnostics import print_diagnostics_report
from plots import (
    plot_shrinkage,
    plot_posterior_comparison,
    plot_spatial_surface,
    plot_anomaly_scores,
    plot_model_comparison_summary,
)


BANNER = """
================================================================
  BAYESIAN REAL ESTATE INTELLIGENCE
  Probabilistic Models for Multi-Portal Market Analysis
================================================================
  Models:
    1. Hierarchical Pricing    (partial pooling)
    2. Gaussian Process        (spatial regression)
    3. Anomaly Detection       (mixture model)
================================================================
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Real Estate Intelligence")
    parser.add_argument("--source", choices=["synthetic", "neon"], default="synthetic")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--n-listings", type=int, default=800)
    parser.add_argument("--no-spatial", action="store_true", help="Skip GP model")
    parser.add_argument("--quick", action="store_true", help="Fewer samples for speed")
    parser.add_argument("--output-dir", default="output", help="Directory for plots")
    return parser.parse_args()


def load_data(args) -> pd.DataFrame:
    if args.source == "neon":
        if not args.database_url:
            print("ERROR: --database-url or DATABASE_URL env var required for neon source")
            sys.exit(1)
        print("Loading from Neon PostgreSQL...")
        df = load_from_neon(args.database_url)
    else:
        print(f"Generating {args.n_listings} synthetic listings...")
        df = generate_synthetic(n_listings=args.n_listings)

    print(f"  Listings: {len(df)}")
    print(f"  Portals:  {df['portal'].nunique()} ({', '.join(df['portal'].unique())})")
    if "zone" in df.columns:
        print(f"  Zones:    {df['zone'].nunique()}")
    print(f"  Price range: {df['price'].min():,.0f} - {df['price'].max():,.0f} EUR")
    return df


def run_hierarchical(df: pd.DataFrame, quick: bool = False) -> HierarchicalPricingModel:
    print("\n" + "=" * 60)
    print(" MODEL 1: HIERARCHICAL PRICING")
    print("=" * 60)

    model = HierarchicalPricingModel(df)
    model.build()

    draws = 500 if quick else 2000
    tune = 300 if quick else 1000
    chains = 2 if quick else 4
    print(f"\n  Sampling NUTS ({draws} draws, {tune} tune, {chains} chains)...")
    model.sample_nuts(draws=draws, tune=tune, chains=chains, cores=1)

    print(f"  NUTS completed in {model.nuts_time:.1f}s")

    print("\n  Sampling ADVI for tractability comparison...")
    model.sample_advi(n_iterations=10000 if quick else 30000)
    print(f"  ADVI completed in {model.advi_time:.1f}s")

    print("\n  Posterior summary:")
    print(model.summary())

    print("\n  Shrinkage analysis:")
    shrinkage = model.shrinkage_summary()
    print(shrinkage.to_string(index=False))

    return model


def run_spatial(df: pd.DataFrame, quick: bool = False) -> SpatialGPModel:
    print("\n" + "=" * 60)
    print(" MODEL 2: GAUSSIAN PROCESS SPATIAL REGRESSION")
    print("=" * 60)

    max_pts = 150 if quick else 300
    model = SpatialGPModel(df, max_points=max_pts)
    model.build()

    draws = 500 if quick else 1000
    print(f"\n  Sampling NUTS ({draws} draws, 2 chains, n={len(model.df)})...")
    print("  (GP models are O(n^3) - this is the tractability bottleneck)")
    model.sample_nuts(draws=draws)
    print(f"  NUTS completed in {model.nuts_time:.1f}s")

    print(f"\n{model.summary()}")
    return model


def run_anomaly(df: pd.DataFrame, quick: bool = False) -> AnomalyMixtureModel:
    print("\n" + "=" * 60)
    print(" MODEL 3: BAYESIAN ANOMALY DETECTION")
    print("=" * 60)

    model = AnomalyMixtureModel(df)
    model.build()

    draws = 500 if quick else 2000
    tune = 300 if quick else 1000
    chains = 2 if quick else 4
    print(f"\n  Sampling NUTS ({draws} draws, {tune} tune, {chains} chains)...")
    model.sample_nuts(draws=draws, tune=tune, chains=chains, cores=1)
    print(f"  NUTS completed in {model.nuts_time:.1f}s")

    print("\n  Sampling ADVI for comparison...")
    model.sample_advi(n_iterations=10000 if quick else 30000)
    print(f"  ADVI completed in {model.advi_time:.1f}s")

    print(f"\n{model.summary()}")

    scores = model.anomaly_scores()
    n_flagged = scores["is_flagged"].sum()
    print(f"\n  Flagged anomalies: {n_flagged}/{len(scores)} ({100*n_flagged/len(scores):.1f}%)")

    if "ground_truth" in scores.columns:
        perf = model.detection_performance()
        print(f"  Detection performance:")
        print(f"    Precision: {perf['precision']:.3f}")
        print(f"    Recall:    {perf['recall']:.3f}")
        print(f"    F1 Score:  {perf['f1_score']:.3f}")

    print("\n  Top 10 most anomalous listings:")
    print(scores.head(10).to_string(index=False))

    return model


def generate_plots(models: dict, df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 60)

    if "hierarchical" in models:
        m = models["hierarchical"]
        shrinkage = m.shrinkage_summary()
        plot_shrinkage(shrinkage, save_path=f"{output_dir}/shrinkage.png")
        plot_posterior_comparison(
            m.trace,
            var_names=["mu_alpha", "mu_beta_size", "sigma"],
            title="Hierarchical Model - Group-Level Posteriors",
            save_path=f"{output_dir}/hierarchical_posteriors.png",
        )
        print("  Saved: shrinkage.png, hierarchical_posteriors.png")

    if "spatial" in models:
        m = models["spatial"]
        try:
            print("  Computing GP prediction grid...")
            grid = m.predict_grid(n_grid=25)
            plot_spatial_surface(grid, df, save_path=f"{output_dir}/spatial_surface.png")
            print("  Saved: spatial_surface.png")
        except Exception as e:
            print(f"  Spatial prediction skipped: {e}")

    if "anomaly" in models:
        m = models["anomaly"]
        scores = m.anomaly_scores()
        plot_anomaly_scores(scores, save_path=f"{output_dir}/anomaly_scores.png")
        print("  Saved: anomaly_scores.png")

    plot_model_comparison_summary(models, save_path=f"{output_dir}/tractability.png")
    print("  Saved: tractability.png")


def main():
    print(BANNER)
    args = parse_args()
    t_total = time.perf_counter()

    # --- Data ---
    df = load_data(args)
    df = preprocess(df)

    # --- Models ---
    models = {}

    models["hierarchical"] = run_hierarchical(df, quick=args.quick)

    if not args.no_spatial and "lat" in df.columns:
        models["spatial"] = run_spatial(df, quick=args.quick)
    elif args.no_spatial:
        print("\n  [Skipping spatial GP model (--no-spatial)]")

    models["anomaly"] = run_anomaly(df, quick=args.quick)

    # --- Diagnostics ---
    traces = {}
    for name, m in models.items():
        if m.trace is not None:
            traces[name] = m.trace

    print_diagnostics_report(models, traces)

    # --- Plots ---
    generate_plots(models, df, args.output_dir)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f" COMPLETE - Total time: {elapsed:.1f}s")
    print(f" Plots saved to: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
