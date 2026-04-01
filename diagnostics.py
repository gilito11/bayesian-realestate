"""
Convergence diagnostics and tractability analysis.

Provides tools to evaluate:
  - MCMC convergence (R-hat, ESS, divergences)
  - Computational efficiency (ESS per second)
  - MCMC vs Variational Inference comparison
  - Model comparison via information criteria (WAIC, LOO-CV)
"""

import numpy as np
import pandas as pd
import arviz as az


def convergence_report(trace: az.InferenceData, model_name: str = "") -> pd.DataFrame:
    """Comprehensive convergence diagnostics for a fitted model."""
    summary = az.summary(trace, kind="diagnostics")
    n_divergent = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        n_divergent = int(trace.sample_stats["diverging"].sum())

    report = {
        "model": model_name,
        "r_hat_max": round(float(summary["r_hat"].max()), 4),
        "r_hat_ok": bool(summary["r_hat"].max() < 1.05),
        "ess_bulk_min": int(summary["ess_bulk"].min()),
        "ess_tail_min": int(summary["ess_tail"].min()),
        "divergences": n_divergent,
        "converged": bool(
            summary["r_hat"].max() < 1.05
            and summary["ess_bulk"].min() > 100
            and n_divergent == 0
        ),
    }
    return pd.DataFrame([report])


def tractability_report(models: dict) -> pd.DataFrame:
    """
    Compare computational cost across models and inference methods.

    Parameters
    ----------
    models : dict
        {name: model_instance} where each instance has .nuts_time
        and optionally .advi_time attributes.
    """
    rows = []
    for name, m in models.items():
        row = {"model": name}

        if hasattr(m, "nuts_time"):
            row["nuts_seconds"] = round(m.nuts_time, 1)
            if m.trace is not None:
                summary = az.summary(m.trace, kind="diagnostics")
                ess_bulk = summary["ess_bulk"].median()
                row["ess_per_second"] = round(ess_bulk / m.nuts_time, 1)

        if hasattr(m, "advi_time"):
            row["advi_seconds"] = round(m.advi_time, 1)

        if hasattr(m, "nuts_time") and hasattr(m, "advi_time"):
            row["speedup_vi"] = round(m.nuts_time / m.advi_time, 1)

        rows.append(row)

    return pd.DataFrame(rows)


def model_comparison(traces: dict) -> pd.DataFrame:
    """
    Compare models using LOO-CV and WAIC.

    Parameters
    ----------
    traces : dict
        {name: az.InferenceData} for each model with log_likelihood group.
    """
    comparison = az.compare(traces, ic="loo")
    return comparison


def print_diagnostics_report(models: dict, traces: dict):
    """Print a formatted report combining all diagnostics."""
    print("\n" + "=" * 60)
    print(" CONVERGENCE DIAGNOSTICS")
    print("=" * 60)

    for name, m in models.items():
        if m.trace is not None:
            report = convergence_report(m.trace, name)
            status = "PASS" if report["converged"].iloc[0] else "FAIL"
            print(f"\n  [{status}] {name}")
            print(f"    R-hat max:     {report['r_hat_max'].iloc[0]}")
            print(f"    ESS bulk min:  {report['ess_bulk_min'].iloc[0]}")
            print(f"    ESS tail min:  {report['ess_tail_min'].iloc[0]}")
            print(f"    Divergences:   {report['divergences'].iloc[0]}")

    print("\n" + "=" * 60)
    print(" TRACTABILITY ANALYSIS")
    print("=" * 60)
    tract = tractability_report(models)
    print(f"\n{tract.to_string(index=False)}")

    if traces:
        try:
            print("\n" + "=" * 60)
            print(" MODEL COMPARISON (LOO-CV)")
            print("=" * 60)
            comp = model_comparison(traces)
            print(f"\n{comp.to_string()}")
        except Exception as e:
            print(f"\n  Could not compute LOO-CV: {e}")
