"""
back.py — Backend module for Customer Segmentation
Handles data loading, KMeans clustering, cluster analysis, and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
N_CLUSTERS = 5
RANDOM_STATE = 42

# Curated cluster colour palette (used by both backend descriptions & frontend)
CLUSTER_COLORS = [
    "#E63946",   # Crimson-red
    "#457B9D",   # Steel-blue
    "#2A9D8F",   # Teal-green
    "#E9C46A",   # Gold
    "#9B5DE5",   # Violet
]

CLUSTER_LABELS = [
    "Careful Spenders",
    "Standard Customers",
    "Target Customers",
    "Careless Spenders",
    "Sensible Savers",
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the Mall Customers dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def get_features(df: pd.DataFrame) -> np.ndarray:
    """Extract Annual Income and Spending Score columns."""
    return df.iloc[:, [3, 4]].values


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(x: np.ndarray, n_clusters: int = N_CLUSTERS):
    """
    Train a KMeans model and return (model, labels).
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(x)
    return kmeans, labels


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def get_cluster_info(x: np.ndarray, labels: np.ndarray, kmeans):
    """
    Return a list of dicts with per-cluster statistics:
      - cluster_id, label, color
      - centroid (income, spending)
      - size (number of customers)
      - avg_income, avg_spending
      - description (human-readable)
    The clusters are sorted by centroid income so that the
    descriptions stay stable across runs.
    """
    info = []
    # Build raw info keyed by cluster id
    raw = {}
    for cid in range(kmeans.n_clusters):
        mask = labels == cid
        pts = x[mask]
        centroid = kmeans.cluster_centers_[cid]
        raw[cid] = {
            "centroid_income": centroid[0],
            "centroid_spending": centroid[1],
            "avg_income": pts[:, 0].mean(),
            "avg_spending": pts[:, 1].mean(),
            "size": int(mask.sum()),
        }

    # Sort cluster ids by centroid income (ascending) for stable ordering
    sorted_ids = sorted(raw.keys(), key=lambda c: (raw[c]["centroid_income"], raw[c]["centroid_spending"]))

    for rank, cid in enumerate(sorted_ids):
        r = raw[cid]
        inc = r["avg_income"]
        spe = r["avg_spending"]

        # Generate a human-readable description
        inc_word = "low" if inc < 40 else ("moderate" if inc < 70 else "high")
        spe_word = "low" if spe < 40 else ("moderate" if spe < 70 else "high")

        if inc_word == "high" and spe_word == "high":
            desc = "High earners who love to spend — your ideal target customers! 🎯"
        elif inc_word == "high" and spe_word == "low":
            desc = "High earners who rarely spend — careful with their money. 💰"
        elif inc_word == "low" and spe_word == "high":
            desc = "Lower income but big spenders — careless with money. 🛒"
        elif inc_word == "low" and spe_word == "low":
            desc = "Lower income and low spending — sensible savers. 🏦"
        else:
            desc = "Average customers who earn and spend moderately. ⚖️"

        info.append({
            "cluster_id": cid,
            "rank": rank,
            "label": _label_for(inc_word, spe_word),
            "color": CLUSTER_COLORS[rank % len(CLUSTER_COLORS)],
            "centroid_income": round(r["centroid_income"], 1),
            "centroid_spending": round(r["centroid_spending"], 1),
            "avg_income": round(inc, 1),
            "avg_spending": round(spe, 1),
            "size": r["size"],
            "income_level": inc_word,
            "spending_level": spe_word,
            "description": desc,
        })

    return info


def _label_for(inc_word: str, spe_word: str) -> str:
    mapping = {
        ("high", "high"): "🎯 Target Customers",
        ("high", "low"): "💰 Careful Spenders",
        ("low", "high"): "🛒 Careless Spenders",
        ("low", "low"): "🏦 Sensible Savers",
    }
    return mapping.get((inc_word, spe_word), "⚖️ Standard Customers")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_customer(kmeans, annual_income: float, spending_score: float):
    """
    Predict which cluster a new customer belongs to.
    Returns the cluster id (int).
    """
    point = np.array([[annual_income, spending_score]])
    return int(kmeans.predict(point)[0])


# ---------------------------------------------------------------------------
# WCSS / Elbow data
# ---------------------------------------------------------------------------

def compute_wcss(x: np.ndarray, max_k: int = 10):
    """Return list of WCSS values for k = 1 … max_k."""
    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE)
        km.fit(x)
        wcss.append(km.inertia_)
    return wcss