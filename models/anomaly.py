"""
Bayesian Anomaly Detection via Mixture Models
==============================================

Identifies anomalous real estate listings by modeling the residual price
distribution as a two-component mixture:

  Component 1 (Normal market):  tight distribution around expected price
  Component 2 (Anomaly):        wide distribution capturing outliers

The posterior probability of belonging to the anomaly component gives
a principled, calibrated anomaly score for each listing — unlike
threshold-based methods, this accounts for parameter uncertainty.

Mathematical formulation:

    Residual model (price after removing size/bedroom effects):
        r_i = log(price_i) - (α + β_size · size_i + β_beds · beds_i)

    Mixture model on residuals:
        w ~ Dirichlet([10, 1])              # prior favors normal component
        z_i ~ Categorical(w)                # latent assignment
        r_i | z_i=0 ~ Normal(μ₁, σ₁)       # normal market
        r_i | z_i=1 ~ Normal(μ₂, σ₂)       # anomaly (σ₂ >> σ₁)
"""

import time
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


class AnomalyMixtureModel:
    """Two-component Bayesian mixture for listing anomaly detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = None
        self.trace = None
        self.residuals = None

    def build(self) -> pm.Model:
        y = self.df["log_price_z"].values
        X_size = self.df["size_m2_z"].values
        X_beds = self.df["bedrooms_z"].values

        with pm.Model() as model:
            # --- First stage: simple regression to get residuals ---
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta_size = pm.Normal("beta_size", mu=0, sigma=3)
            beta_beds = pm.Normal("beta_beds", mu=0, sigma=3)
            regression_mu = alpha + beta_size * X_size + beta_beds * X_beds

            # --- Mixture on residuals ---
            w = pm.Dirichlet("w", a=np.array([10.0, 1.0]))

            # Normal market component (tight)
            mu_normal = pm.Normal("mu_normal", mu=0, sigma=0.5)
            sigma_normal = pm.HalfNormal("sigma_normal", sigma=0.5)

            # Anomaly component (wide)
            mu_anomaly = pm.Normal("mu_anomaly", mu=0, sigma=2)
            sigma_anomaly = pm.HalfNormal("sigma_anomaly", sigma=3)

            pm.Mixture(
                "likelihood",
                w=w,
                comp_dists=[
                    pm.Normal.dist(mu=regression_mu + mu_normal, sigma=sigma_normal),
                    pm.Normal.dist(mu=regression_mu + mu_anomaly, sigma=sigma_anomaly),
                ],
                observed=y,
            )

        self.model = model
        return model

    def sample_nuts(self, draws=2000, tune=1000, chains=4, cores=1, seed=42) -> az.InferenceData:
        t0 = time.perf_counter()
        with self.model:
            self.trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                random_seed=seed, return_inferencedata=True,
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

    def anomaly_scores(self) -> pd.DataFrame:
        """
        Compute posterior anomaly probability for each listing.

        Uses the mixture weights and component likelihoods to estimate
        P(z_i = anomaly | data) for each observation.
        """
        post = self.trace.posterior
        y = self.df["log_price_z"].values
        X_size = self.df["size_m2_z"].values
        X_beds = self.df["bedrooms_z"].values

        # Extract posterior means
        alpha = float(post["alpha"].mean())
        beta_s = float(post["beta_size"].mean())
        beta_b = float(post["beta_beds"].mean())
        reg_mu = alpha + beta_s * X_size + beta_b * X_beds

        w = post["w"].mean(dim=["chain", "draw"]).values
        mu_n = float(post["mu_normal"].mean())
        sigma_n = float(post["sigma_normal"].mean())
        mu_a = float(post["mu_anomaly"].mean())
        sigma_a = float(post["sigma_anomaly"].mean())

        from scipy import stats
        residuals = y - reg_mu
        ll_normal = stats.norm.pdf(residuals, loc=mu_n, scale=sigma_n)
        ll_anomaly = stats.norm.pdf(residuals, loc=mu_a, scale=sigma_a)

        p_anomaly = (w[1] * ll_anomaly) / (w[0] * ll_normal + w[1] * ll_anomaly + 1e-12)

        result = self.df[["listing_id", "portal", "zone", "price", "size_m2"]].copy()
        result["residual"] = residuals
        result["p_anomaly"] = p_anomaly
        result["is_flagged"] = p_anomaly > 0.5
        if "is_anomaly" in self.df.columns:
            result["ground_truth"] = self.df["is_anomaly"]
        return result.sort_values("p_anomaly", ascending=False)

    def detection_performance(self) -> dict:
        """Evaluate anomaly detection against known ground truth (synthetic data)."""
        scores = self.anomaly_scores()
        if "ground_truth" not in scores.columns:
            return {"note": "No ground truth available (real data)"}

        tp = ((scores["is_flagged"]) & (scores["ground_truth"])).sum()
        fp = ((scores["is_flagged"]) & (~scores["ground_truth"])).sum()
        fn = ((~scores["is_flagged"]) & (scores["ground_truth"])).sum()
        tn = ((~scores["is_flagged"]) & (~scores["ground_truth"])).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        }

    def summary(self) -> str:
        return az.summary(self.trace, var_names=[
            "w", "alpha", "beta_size", "beta_beds",
            "mu_normal", "sigma_normal", "mu_anomaly", "sigma_anomaly",
        ]).to_string()
