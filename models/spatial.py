"""
Gaussian Process Spatial Price Model
=====================================

Models real estate prices as a smooth function of geographic coordinates
using a Gaussian Process with a Matern-5/2 covariance kernel.

This captures spatial price patterns - proximity to coastline, city centers,
transport - without explicitly encoding those features. The GP learns the
spatial structure from data alone, producing:

  1. A continuous price surface over the geographic region
  2. Calibrated uncertainty that grows in data-sparse areas
  3. Automatic detection of price gradients and local effects

Mathematical formulation:

    f ~ GP(0, k(x, x'))
    k(x, x') = eta^2 * Matern_5/2(|x - x'| / l)

    y_i ~ Normal(f(x_i) + X_i * beta, sigma)

where x_i = (lat, lon) and X_i are property features.
"""

import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


class SpatialGPModel:
    """GP regression for spatial price surfaces with uncertainty."""

    def __init__(self, df: pd.DataFrame, max_points: int = 300):
        # GP scales as O(n^3), so we subsample for tractability
        if len(df) > max_points:
            self.df = df.sample(n=max_points, random_state=42).reset_index(drop=True)
            self.subsampled = True
        else:
            self.df = df.copy()
            self.subsampled = False

        self.model = None
        self.trace = None
        self.gp = None

    def build(self) -> pm.Model:
        coords = self.df[["lat", "lon"]].values
        # Normalize coordinates to ~unit scale for kernel
        self.coord_mean = coords.mean(axis=0)
        self.coord_std = coords.std(axis=0) + 1e-6
        X_spatial = (coords - self.coord_mean) / self.coord_std

        X_size = self.df["size_m2_z"].values
        y = self.df["log_price_z"].values

        with pm.Model() as model:
            # Feature coefficient (non-spatial)
            beta_size = pm.Normal("beta_size", mu=0, sigma=2)
            feature_effect = beta_size * X_size

            # GP hyperparameters
            lengthscale = pm.Gamma("lengthscale", alpha=2, beta=2)
            amplitude = pm.HalfNormal("amplitude", sigma=2)

            # Matern-5/2 covariance (smoother than 3/2, more flexible than RBF)
            cov = amplitude**2 * pm.gp.cov.Matern52(input_dim=2, ls=lengthscale)
            self.gp = pm.gp.Marginal(cov_func=cov)

            # Noise
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Marginal likelihood (analytically integrates out GP values)
            self.gp.marginal_likelihood(
                "likelihood",
                X=X_spatial,
                y=y - feature_effect,
                sigma=sigma,
            )

        self.X_spatial = X_spatial
        self.y_residual = y - feature_effect
        self.model = model
        return model

    def sample_nuts(self, draws=1000, tune=1000, chains=2, seed=42) -> az.InferenceData:
        """Fewer chains/draws than hierarchical due to GP computational cost."""
        t0 = time.perf_counter()
        with self.model:
            self.trace = pm.sample(
                draws=draws, tune=tune, chains=chains,
                random_seed=seed, return_inferencedata=True,
                target_accept=0.90,
            )
        self.nuts_time = time.perf_counter() - t0
        return self.trace

    def predict_grid(self, n_grid: int = 30) -> dict:
        """Predict prices on a lat/lon grid for surface visualization."""
        lat_range = np.linspace(self.df["lat"].min(), self.df["lat"].max(), n_grid)
        lon_range = np.linspace(self.df["lon"].min(), self.df["lon"].max(), n_grid)
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
        X_new = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        X_new_norm = (X_new - self.coord_mean) / self.coord_std

        with self.model:
            pred = self.gp.conditional("f_pred", X_new_norm)
            post_pred = pm.sample_posterior_predictive(
                self.trace, var_names=["f_pred"], random_seed=42,
            )

        f_samples = post_pred.posterior_predictive["f_pred"].values
        f_samples = f_samples.reshape(-1, n_grid * n_grid)

        return {
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
            "f_mean": f_samples.mean(axis=0).reshape(n_grid, n_grid),
            "f_std": f_samples.std(axis=0).reshape(n_grid, n_grid),
        }

    def summary(self) -> str:
        info = f"Spatial GP Model (n={len(self.df)}"
        if self.subsampled:
            info += ", subsampled for tractability"
        info += ")\n"
        info += az.summary(self.trace, var_names=[
            "lengthscale", "amplitude", "beta_size", "sigma"
        ]).to_string()
        return info
