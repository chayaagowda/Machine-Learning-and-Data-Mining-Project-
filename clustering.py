"""
Clustering — Student Risk Profiling (Unsupervised Learning)
ITMD 522 | Joshua Godwin

Implements:
  - K-Means (K swept 2..10, optimal K via Elbow + Silhouette)
  - Agglomerative Hierarchical Clustering (Ward linkage) at the same K
  - Cluster profiling on key features (GPA, LMS engagement, credits, scholarship, etc.)
  - Cluster evaluation: Silhouette Score, Davies-Bouldin Index, Cluster Purity
  - Cross-references clusters with the actual dropout label (label is NOT used during clustering)

Memory-efficient design for CPU-only machines:
  - Stratified subsample of CLUSTER_SAMPLE rows from the merged 464k dataset
  - Same preprocessing pipeline as classification.py for consistency

Outputs (under results/clustering/):
  - elbow_silhouette.png         : Elbow + silhouette diagnostic plots
  - dendrogram.png               : Ward linkage dendrogram (truncated)
  - kmeans_pca_scatter.png       : 2-D PCA scatter colored by K-Means cluster
  - agg_pca_scatter.png          : 2-D PCA scatter colored by Agglomerative cluster
  - cluster_profiles_kmeans.csv  : per-cluster mean of key features + dropout rate
  - cluster_profiles_agg.csv     : same, for Agglomerative
  - clustering_summary.csv       : Silhouette, Davies-Bouldin, Purity for both methods
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "clustering")
TARGET_COL      = "abandono_hash"
RANDOM_STATE    = 42
CLUSTER_SAMPLE  = 10_000        # stratified rows (kept on CPU for K-Means + Ward)
AGG_SAMPLE      = 5_000         # Agglomerative is O(n^2) — keep it smaller
K_RANGE         = range(2, 11)  # K = 2..10 for sweep
OUTLIER_QUANTILE = 0.99         # cap each clustering feature at this quantile
                                # to keep Ward from isolating extreme outliers

os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = [
    "dni_hash", "tit_hash", "asi_hash",
    "grupos_por_tipocredito_hash",
    "baja_fecha", "fecha_datos", "caca",
]

COMMA_DECIMAL_COLS = [
    "nota10_hash", "nota14_hash", "nota_asig_hash", "matricula_activa",
]

LMS_PREFIXES = [
    "pft_events", "pft_days_logged", "pft_visits",
    "pft_assignment_submissions", "pft_test_submissions", "pft_total_minutes",
    "n_wifi_days", "resource_events", "n_resource_days",
]

MONTHLY_RE = re.compile(r".+_\d{4}_\d+$")

# Curated numeric feature set used BOTH for clustering and for profiling.
# Using just these instead of the full 100+ one-hot space avoids the high-dimensionality
# breakdown of distance-based clustering and produces interpretable cluster profiles.
PROFILE_FEATURES = [
    "nota10_hash",                       # GPA-like grade (out of 10)
    "rendimiento_total",                 # academic progression rate
    "cred_sup_total",                    # total credits passed
    "cred_mat_total",                    # total credits attempted
    "total_pft_events",                  # total LMS events
    "total_pft_visits",                  # total LMS visits
    "total_pft_assignment_submissions",
    "total_n_wifi_days",                 # total Wi-Fi days (campus presence)
    "total_resource_events",             # total LMS resource accesses
    "anyo_ingreso",                      # year of enrollment
]


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{DATA_DIR}'.")

    frames = []
    for fname in csv_files:
        df = pd.read_csv(
            os.path.join(DATA_DIR, fname),
            sep=";", low_memory=False, on_bad_lines="skip"
        )
        print(f"  Loaded {fname}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {combined.shape[0]:,} rows x {combined.shape[1]} cols")
    return combined


# ─────────────────────────────────────────────
# 2. PREPROCESS  (mirror of classification.py preprocessing)
# ─────────────────────────────────────────────
def preprocess(df):
    # Encode dropout label (NOT used in clustering — only in evaluation/profiling)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df["dropout"] = (df[TARGET_COL] == "A").astype(int)
    print(f"\n  Class distribution: {df['dropout'].value_counts().to_dict()}")

    df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns],
            inplace=True)

    # Comma-as-decimal
    for col in COMMA_DECIMAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    all_monthly = [c for c in df.columns if MONTHLY_RE.match(c)]
    for col in all_monthly:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # Aggregate monthly LMS / Wi-Fi columns into totals
    for prefix in LMS_PREFIXES:
        pat = re.compile(rf"^{re.escape(prefix)}_\d{{4}}_\d+$")
        month_cols = [c for c in df.columns if pat.match(c)]
        if month_cols:
            df[f"total_{prefix}"] = df[month_cols].fillna(0).sum(axis=1)
            df.drop(columns=month_cols, inplace=True)

    leftover = [c for c in df.columns if MONTHLY_RE.match(c)]
    if leftover:
        df.drop(columns=leftover, inplace=True)

    # Separate label so it never leaks into clustering features
    y = df.pop("dropout")

    # Keep ONLY the curated numeric feature set for clustering (no one-hot dummies).
    available = [c for c in PROFILE_FEATURES if c in df.columns]
    X_cluster = df[available].copy()
    print(f"  Clustering on {len(available)} curated numeric features: {available}")

    # Force every curated column to numeric — some columns in the source CSV use
    # comma-as-decimal (e.g. rendimiento_total, cred_sup_total) and aren't in the
    # explicit COMMA_DECIMAL_COLS list. Coerce errors to NaN, then median-impute.
    for col in X_cluster.columns:
        if X_cluster[col].dtype == object:
            X_cluster[col] = pd.to_numeric(
                X_cluster[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
        else:
            X_cluster[col] = pd.to_numeric(X_cluster[col], errors="coerce")

    # Numeric impute (median)
    X_cluster = X_cluster.fillna(X_cluster.median(numeric_only=True))
    # Drop any column that is still all-NaN (no useful data)
    all_nan = [c for c in X_cluster.columns if X_cluster[c].isna().all()]
    if all_nan:
        X_cluster.drop(columns=all_nan, inplace=True)
        print(f"  Dropped all-NaN columns: {all_nan}")

    # Outlier capping at OUTLIER_QUANTILE per feature, to keep Ward linkage from
    # isolating a single extreme observation as its own cluster.
    caps = X_cluster.quantile(OUTLIER_QUANTILE)
    X_cluster = X_cluster.clip(upper=caps, axis=1)
    print(f"  Capped each feature at the {int(OUTLIER_QUANTILE*100)}th percentile")

    # Stratified subsample (stratify on dropout to preserve class ratio for purity calc)
    if CLUSTER_SAMPLE and len(X_cluster) > CLUSTER_SAMPLE:
        X_sub, _, y_sub, _ = train_test_split(
            X_cluster, y,
            train_size=CLUSTER_SAMPLE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"  Subsampled to {CLUSTER_SAMPLE:,} rows for clustering "
              f"(positive rate: {y_sub.mean():.3f})")
        X_cluster, y = X_sub, y_sub

    return X_cluster.reset_index(drop=True), y.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. ELBOW + SILHOUETTE SWEEP
# ─────────────────────────────────────────────
def elbow_silhouette_sweep(X_scaled):
    inertias, silhouettes = [], []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # Silhouette on a sample for speed if X is large
        sample_size = min(3000, len(X_scaled))
        s = silhouette_score(X_scaled, labels, sample_size=sample_size,
                             random_state=RANDOM_STATE)
        silhouettes.append(s)
        print(f"    K={k:2d}  inertia={km.inertia_:>12,.0f}  silhouette={s:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(list(K_RANGE), inertias, "o-", color="steelblue")
    ax1.set_xlabel("Number of clusters K")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.grid(alpha=0.3)

    ax2.plot(list(K_RANGE), silhouettes, "o-", color="darkorange")
    ax2.set_xlabel("Number of clusters K")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "elbow_silhouette.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)

    # Pick K with the highest silhouette as the optimal K
    optimal_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"\n  Optimal K (max silhouette): {optimal_k}")
    return optimal_k, inertias, silhouettes


# ─────────────────────────────────────────────
# 4. CLUSTER PURITY
# ─────────────────────────────────────────────
def cluster_purity(labels, y_true):
    """Proportion of records belonging to the majority dropout class within each cluster,
    averaged (weighted) across clusters."""
    df = pd.DataFrame({"cluster": labels, "y": y_true})
    correct = 0
    for _, group in df.groupby("cluster"):
        correct += group["y"].value_counts().iloc[0]
    return correct / len(df)


# ─────────────────────────────────────────────
# 5. CLUSTER PROFILING
# ─────────────────────────────────────────────
def profile_clusters(X_raw, labels, y, name):
    """Compute mean of key features + dropout rate per cluster."""
    available = [c for c in PROFILE_FEATURES if c in X_raw.columns]
    profile = X_raw[available].copy()
    profile["cluster"] = labels
    profile["dropout"] = y.values

    agg_dict = {c: "mean" for c in available}
    agg_dict["dropout"] = "mean"   # = dropout rate within cluster
    summary = profile.groupby("cluster").agg(agg_dict).round(3)
    summary["size"] = profile.groupby("cluster").size()
    summary["dropout_rate"] = summary.pop("dropout")

    out = os.path.join(OUTPUT_DIR, f"cluster_profiles_{name}.csv")
    summary.to_csv(out)
    print(f"\n  {name.upper()} cluster profiles saved -> {out}")
    print(summary.to_string())
    return summary


# ─────────────────────────────────────────────
# 6. PCA SCATTER
# ─────────────────────────────────────────────
def pca_scatter(X_scaled, labels, title, fname):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                    s=8, alpha=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Cluster")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────
# 7. DENDROGRAM
# ─────────────────────────────────────────────
def plot_dendrogram(X_scaled_sample):
    Z = linkage(X_scaled_sample, method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=9, show_contracted=True, ax=ax)
    ax.set_title("Agglomerative Hierarchical Clustering — Ward Linkage (truncated)")
    ax.set_xlabel("Cluster size (or sample index)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dendrogram.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STUDENT RISK PROFILING — CLUSTERING")
    print("=" * 60)

    print("\n[1] Loading data ...")
    df = load_data()

    print("\n[2] Preprocessing ...")
    X_raw, y = preprocess(df)

    print("\n[3] Scaling ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"  Scaled feature matrix: {X_scaled.shape}")

    print("\n[4] Elbow + Silhouette sweep (K-Means K=2..10) ...")
    optimal_k, _, _ = elbow_silhouette_sweep(X_scaled)

    summary_rows = []

    # ---- K-Means at optimal K ----
    print(f"\n[5] Fitting final K-Means with K={optimal_k} ...")
    km_final = KMeans(n_clusters=optimal_k, n_init=20, random_state=RANDOM_STATE)
    km_labels = km_final.fit_predict(X_scaled)

    sample_size = min(3000, len(X_scaled))
    km_sil = silhouette_score(X_scaled, km_labels, sample_size=sample_size,
                              random_state=RANDOM_STATE)
    km_dbi = davies_bouldin_score(X_scaled, km_labels)
    km_pur = cluster_purity(km_labels, y)
    print(f"  K-Means | Silhouette={km_sil:.4f} | DBI={km_dbi:.4f} | Purity={km_pur:.4f}")

    profile_clusters(X_raw, km_labels, y, "kmeans")
    pca_scatter(X_scaled, km_labels,
                f"K-Means clusters (K={optimal_k}) — 2-D PCA projection",
                "kmeans_pca_scatter.png")

    summary_rows.append({
        "Method": "K-Means", "K": optimal_k,
        "Silhouette": round(km_sil, 4),
        "Davies_Bouldin": round(km_dbi, 4),
        "Purity": round(km_pur, 4),
        "n_samples": len(X_scaled),
    })

    # ---- Agglomerative (Ward) at the same K ----
    print(f"\n[6] Fitting Agglomerative (Ward) with K={optimal_k} ...")
    if len(X_scaled) > AGG_SAMPLE:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_scaled), size=AGG_SAMPLE, replace=False
        )
        X_agg = X_scaled[idx]
        y_agg = y.iloc[idx].reset_index(drop=True)
        X_raw_agg = X_raw.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {AGG_SAMPLE:,} rows for Ward (O(n^2) memory).")
    else:
        X_agg, y_agg, X_raw_agg = X_scaled, y, X_raw

    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    agg_labels = agg.fit_predict(X_agg)

    agg_sil = silhouette_score(X_agg, agg_labels,
                               sample_size=min(3000, len(X_agg)),
                               random_state=RANDOM_STATE)
    agg_dbi = davies_bouldin_score(X_agg, agg_labels)
    agg_pur = cluster_purity(agg_labels, y_agg)
    print(f"  Agglomerative | Silhouette={agg_sil:.4f} | DBI={agg_dbi:.4f} | Purity={agg_pur:.4f}")

    profile_clusters(X_raw_agg, agg_labels, y_agg, "agg")
    pca_scatter(X_agg, agg_labels,
                f"Agglomerative clusters (K={optimal_k}) — 2-D PCA projection",
                "agg_pca_scatter.png")

    summary_rows.append({
        "Method": "Agglomerative (Ward)", "K": optimal_k,
        "Silhouette": round(agg_sil, 4),
        "Davies_Bouldin": round(agg_dbi, 4),
        "Purity": round(agg_pur, 4),
        "n_samples": len(X_agg),
    })

    print(f"\n[7] Dendrogram on a 1500-row Ward sub-sample ...")
    dendro_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_scaled), size=min(1500, len(X_scaled)), replace=False
    )
    plot_dendrogram(X_scaled[dendro_idx])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "clustering_summary.csv"),
                      index=False)

    print("\n" + "=" * 60)
    print("  CLUSTERING SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n  Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
