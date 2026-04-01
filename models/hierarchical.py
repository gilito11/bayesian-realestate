"""
Hierarchical Bayesian Pricing Model
====================================

Multi-level partial pooling across real estate portals.

Each portal has its own intercept and feature coefficients, drawn from
a shared group-level distribution. This enables information sharing:
portals with few listings borrow strength from portals with many,
while portals with enough data can diverge from the group mean.

Mathematical formulation:

    Group level:
        μ_α ~ Normal(0, 10)        σ_α ~ HalfNormal(5)
        μ_β ~ Normal(0, 5)         σ_β ~ HalfNormal(3)

    Portal level (j = 1..J):
        α_j ~ Normal(μ_α, σ_α)
        β_j ~ Normal(μ_β, σ_β)     [vector for each feature]

    Observation level (i = 1..N):
        y_i ~ Normal(α_{j[i]} + X_i · β_{j[i]}, σ)
"""

import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


class HierarchicalPricingModel:
    """Hierarchical Bayesian model with portal-level partial pooling."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.portal_idx = df["portal_idx"].values
        self.n_portals = df["portal_idx"].nunique()
        self.portal_names = df["portal"].unique().tolist()
        self.model = None
        self.trace = None
        self.trace_vi = None

    def build(self) -> pm.Model:
        X_size = self.df["size_m2_z"].values
        X_beds = self.df["bedrooms_z"].values
        X_baths = self.df["bathrooms_z"].values
        y = self.df["log_price_z"].values

        with pm.Model() as model:
            # --- Hyperpriors (group-level) ---
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5)

            mu_beta_size = pm.Normal("mu_beta_size", mu=0, sigma=5)
            sigma_beta_size = pm.HalfNormal("sigma_beta_size", sigma=3)

            mu_beta_beds = pm.Normal("mu_beta_beds", mu=0, sigma=5)
            sigma_beta_beds = pm.HalfNormal("sigma_beta_beds", sigma=3)

            mu_beta_baths = pm.Normal("mu_beta_baths", mu=0, sigma=5)
            sigma_beta_baths = pm.HalfNormal("sigma_beta_baths", sigma=3)

            # --- Portal-level (partial pooling via shared priors) ---
            alpha = pm.Normal(
                "alpha", mu=mu_alpha, sigma=sigma_alpha, shape=self.n_portals
            )
            beta_size = pm.Normal(
                "beta_size", mu=mu_beta_size, sigma=sigma_beta_size, shape=self.n_portals
            )
            beta_beds = pm.Normal(
                "beta_beds", mu=mu_beta_beds, sigma=sigma_beta_beds, shape=self.n_portals
            )
            beta_baths = pm.Normal(
                "beta_baths", mu=mu_beta_baths, sigma=sigma_beta_baths, shape=self.n_portals
            )

            # --- Observation model ---
            sigma = pm.HalfNormal("sigma", sigma=5)

            mu = (
                alpha[self.portal_idx]
                + beta_size[self.portal_idx] * X_size
                + beta_beds[self.portal_idx] * X_beds
                + beta_baths[self.portal_idx] * X_baths
            )

            pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        self.model = model
        return model

    def sample_nuts(self, draws=2000, tune=1000, chains=4, cores=1, seed=42) -> az.InferenceData:
        t0 = time.perf_counter()
        with self.model:
            self.trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                random_seed=seed, return_inferencedata=True,
                progressbar=True,
            )
        self.nuts_time = time.perf_counter() - t0
        return self.trace

    def sample_advi(self, n_iterations=30000, seed=42) -> az.InferenceData:
        t0 = time.perf_counter()
        with self.model:
            approx = pm.fit(n=n_iterations, method="advi", random_seed=seed)
            self.trace_vi = approx.sample(2000)
        self.advi_time = time.perf_counter() - t0
        return self.trace_vi

    def shrinkage_summary(self) -> pd.DataFrame:
        """Show how portal estimates shrink toward the group mean."""
        post = self.trace.posterior
        rows = []
        for j, portal in enumerate(self.portal_names):
            rows.append({
                "portal": portal,
                "alpha_mean": float(post["alpha"].sel(alpha_dim_0=j).mean()),
                "alpha_hdi_low": float(az.hdi(post["alpha"].sel(alpha_dim_0=j).values.flatten(), hdi_prob=0.94)[0]),
                "alpha_hdi_high": float(az.hdi(post["alpha"].sel(alpha_dim_0=j).values.flatten(), hdi_prob=0.94)[1]),
                "beta_size_mean": float(post["beta_size"].sel(beta_size_dim_0=j).mean()),
                "group_mu_alpha": float(post["mu_alpha"].mean()),
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        return az.summary(self.trace, var_names=[
            "mu_alpha", "sigma_alpha",
            "mu_beta_size", "mu_beta_beds", "mu_beta_baths",
            "alpha", "beta_size", "sigma",
        ]).to_string()
