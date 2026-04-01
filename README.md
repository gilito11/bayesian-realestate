# Bayesian Real Estate Intelligence

**Probabilistic modeling for multi-portal real estate market analysis**

A probabilistic programming framework that applies hierarchical Bayesian models, Gaussian Process spatial regression, and mixture-model anomaly detection to real estate listing data from four Spanish portals. Built with [PyMC](https://www.pymc.io/) and [ArviZ](https://arviz-devs.github.io/arviz/).

> Demo project for IAESTE internship application — Materials Center Leoben (AT-2026-4006LE)

---

## Key Features

| Model | Technique | Purpose |
|-------|-----------|---------|
| **Hierarchical Pricing** | Multi-level partial pooling | Share information across portals while respecting per-portal price dynamics |
| **Spatial GP Regression** | Gaussian Process with Matérn-5/2 kernel | Continuous price surfaces with calibrated uncertainty over geographic zones |
| **Anomaly Detection** | Bayesian mixture model | Identify overpriced/underpriced listings via latent component membership |

### Tractability Analysis

Each model includes a full tractability report:
- **MCMC vs Variational Inference** comparison (NUTS vs ADVI)
- **Effective Sample Size per second** (ESS/s) as efficiency metric
- **R-hat convergence** diagnostics
- **Model comparison** via WAIC and LOO-CV

---

## Architecture

```
bayesian_realestate/
├── models/
│   ├── hierarchical.py   # Hierarchical multi-portal pricing
│   ├── spatial.py         # Gaussian Process spatial regression
│   └── anomaly.py         # Mixture model anomaly detection
├── data.py                # Synthetic data generation + Neon DB loader
├── diagnostics.py         # Convergence & tractability analysis
├── plots.py               # ArviZ + custom visualizations
└── demo.py                # Full pipeline demo
```

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

With real data from Neon PostgreSQL:
```bash
python demo.py --source neon --database-url $DATABASE_URL
```

## Technical Stack

- **PPL**: PyMC 5.x (NUTS sampler, ADVI variational inference)
- **Diagnostics**: ArviZ (posterior analysis, model comparison)
- **Data**: pandas, NumPy, optional Neon PostgreSQL via psycopg2
- **Visualization**: matplotlib, seaborn

## Data

The framework works in two modes:

1. **Synthetic** (default): Generates realistic multi-portal listings with known ground truth — useful for validating model recovery
2. **Neon PostgreSQL**: Connects to a live database of scraped listings from habitaclia, fotocasa, milanuncios, and idealista

## Example Output

```
══════════════════════════════════════════════════════
 BAYESIAN REAL ESTATE INTELLIGENCE — Model Summary
══════════════════════════════════════════════════════

 Model Comparison (LOO-CV)
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Model               │ elpd_loo │ p_loo    │ Rank     │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Hierarchical        │ -342.1   │ 12.3     │ 1        │
│ Spatial GP          │ -358.7   │ 8.1      │ 2        │
│ Pooled Baseline     │ -401.2   │ 4.0      │ 3        │
└─────────────────────┴──────────┴──────────┴──────────┘

 Tractability Report
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Model               │ NUTS (s) │ ADVI (s) │ ESS/s    │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Hierarchical        │ 45.2     │ 8.1      │ 89.3     │
│ Spatial GP          │ 182.4    │ 22.3     │ 21.7     │
│ Anomaly Mixture     │ 38.7     │ 12.4     │ 103.2    │
└─────────────────────┴──────────┴──────────┴──────────┘
```

## Author

**Eric Gil** — Computer Science, Universitat de Lleida
