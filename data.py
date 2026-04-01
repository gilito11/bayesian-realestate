"""
Data generation and loading for Bayesian Real Estate Intelligence.

Two modes:
  - Synthetic: reproducible fake data with known ground truth
  - Neon: live data from scraped Spanish real estate portals
"""

import numpy as np
import pandas as pd


# -- Zone definitions with approximate lat/lon centers (Tarragona coast) ------

ZONES = {
    "salou":      (41.076, 1.142),
    "cambrils":   (41.067, 1.060),
    "tarragona":  (41.119, 1.245),
    "reus":       (41.155, 1.106),
    "vila-seca":  (41.112, 1.147),
    "torredembarra": (41.146, 1.396),
    "altafulla":  (41.143, 1.378),
    "la-pineda":  (41.095, 1.170),
}

PORTALS = ["habitaclia", "fotocasa", "milanuncios", "idealista"]

PROPERTY_TYPES = ["piso", "casa", "duplex", "atico", "estudio"]


def generate_synthetic(
    n_listings: int = 800,
    n_anomalies: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic multi-portal listing data with injected anomalies."""
    rng = np.random.default_rng(seed)

    zones = list(ZONES.keys())
    zone_base_price = {
        "salou": 11.2, "cambrils": 11.0, "tarragona": 11.3,
        "reus": 10.6, "vila-seca": 11.0, "torredembarra": 11.1,
        "altafulla": 11.15, "la-pineda": 10.9,
    }
    portal_bias = {
        "habitaclia": 0.0, "fotocasa": 0.03,
        "milanuncios": -0.05, "idealista": 0.02,
    }

    records = []
    for _ in range(n_listings):
        zone = rng.choice(zones)
        portal = rng.choice(PORTALS)
        lat, lon = ZONES[zone]
        lat += rng.normal(0, 0.01)
        lon += rng.normal(0, 0.01)

        prop_type = rng.choice(PROPERTY_TYPES, p=[0.45, 0.25, 0.10, 0.10, 0.10])
        bedrooms = rng.choice([1, 2, 3, 4], p=[0.15, 0.40, 0.30, 0.15])
        bathrooms = rng.choice([1, 2, 3], p=[0.50, 0.40, 0.10])
        size_m2 = max(25, int(rng.normal(70 + bedrooms * 20, 20)))

        type_premium = {"piso": 0, "casa": 0.15, "duplex": 0.08, "atico": 0.12, "estudio": -0.10}
        log_price = (
            zone_base_price[zone]
            + portal_bias[portal]
            + 0.006 * size_m2
            + 0.05 * bedrooms
            + 0.04 * bathrooms
            + type_premium[prop_type]
            + rng.normal(0, 0.12)
        )

        days_ago = rng.integers(0, 180)
        date = pd.Timestamp("2026-04-01") - pd.Timedelta(days=int(days_ago))
        trend = -0.0003 * days_ago  # slight downward trend over time
        log_price += trend

        records.append({
            "portal": portal,
            "zone": zone,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "property_type": prop_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "size_m2": size_m2,
            "log_price": round(log_price, 4),
            "price": round(np.exp(log_price)),
            "date": date,
            "is_anomaly": False,
        })

    # Inject anomalies (extreme under/overpricing)
    anomaly_idx = rng.choice(len(records), size=n_anomalies, replace=False)
    for idx in anomaly_idx:
        direction = rng.choice([-1, 1])
        records[idx]["log_price"] += direction * rng.uniform(0.5, 1.2)
        records[idx]["price"] = round(np.exp(records[idx]["log_price"]))
        records[idx]["is_anomaly"] = True

    df = pd.DataFrame(records)
    df["listing_id"] = [f"SYN-{i:04d}" for i in range(len(df))]
    return df.sort_values("date").reset_index(drop=True)


def load_from_neon(database_url: str) -> pd.DataFrame:
    """Load real listings from Neon PostgreSQL (Casa Teva CRM)."""
    import psycopg2

    query = """
        SELECT
            portal,
            location AS zone,
            property_type,
            bedrooms,
            bathrooms,
            size_m2,
            price,
            ln(price) AS log_price,
            created_at AS date
        FROM raw.raw_listings
        WHERE price > 10000
          AND price < 2000000
          AND size_m2 > 10
          AND size_m2 < 1000
        ORDER BY created_at DESC
        LIMIT 2000
    """
    with psycopg2.connect(database_url) as conn:
        df = pd.read_sql(query, conn)

    df["is_anomaly"] = False  # unknown for real data
    df["listing_id"] = df.index.astype(str)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize features for modeling."""
    df = df.copy()
    for col in ["size_m2", "bedrooms", "bathrooms"]:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mean) / std
    df["log_price_z"] = (df["log_price"] - df["log_price"].mean()) / df["log_price"].std()

    portal_codes, portal_uniques = pd.factorize(df["portal"])
    df["portal_idx"] = portal_codes
    df.attrs["portal_names"] = list(portal_uniques)

    zone_codes, zone_uniques = pd.factorize(df["zone"])
    df["zone_idx"] = zone_codes
    df.attrs["zone_names"] = list(zone_uniques)

    if "date" in df.columns:
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    return df
