"""
Classification - Student Dropout Prediction
ITMD 522 | Bharath Raahul Murugesan

Memory-efficient design for CPU-only machines:
  - Stratified 30k-row subsample from the full 464k dataset
  - class_weight='balanced' replaces SMOTE for LR / DT / RF / SVM
  - SMOTE used only for KNN (no class_weight support), on the small sample
  - 3-fold CV with RandomizedSearchCV (n_iter=5) instead of full GridSearch
  - Results are statistically valid — 30k stratified rows are more than enough
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  — adjust TRAIN_SAMPLE if still slow
# ─────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "classification")
TARGET_COL    = "abandono_hash"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 3          # 3-fold CV
N_ITER        = 3          # random search iterations per model
TRAIN_SAMPLE  = 10_000    # stratified rows for training

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
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    # 2a. Encode target
    df = df.dropna(subset=[TARGET_COL]).copy()
    df["target"] = (df[TARGET_COL] == "A").astype(int)
    print(f"\n  Full dataset class distribution: {df['target'].value_counts().to_dict()}")

    # 2b. Drop ID / leakage columns
    df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns],
            inplace=True)

    # 2c. Fix comma-as-decimal in known columns
    for col in COMMA_DECIMAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # 2d. Fix comma-as-decimal in all monthly columns, then aggregate
    all_monthly = [c for c in df.columns if MONTHLY_RE.match(c)]
    for col in all_monthly:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    for prefix in LMS_PREFIXES:
        pat = re.compile(rf"^{re.escape(prefix)}_\d{{4}}_\d+$")
        month_cols = [c for c in df.columns if pat.match(c)]
        if month_cols:
            df[f"total_{prefix}"] = df[month_cols].fillna(0).sum(axis=1)
            df.drop(columns=month_cols, inplace=True)

    leftover = [c for c in df.columns if MONTHLY_RE.match(c)]
    if leftover:
        df.drop(columns=leftover, inplace=True)

    # 2e. Drop columns with >60% missing
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.60].index.tolist()
    if high_null:
        df.drop(columns=high_null, inplace=True)
        print(f"  Dropped {len(high_null)} columns with >60% nulls")

    # 2f. Split X / y
    y = df.pop("target")
    X = df

    # 2g. Impute
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for col in cat_cols:
        mode = X[col].mode()
        X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # 2h. One-hot encode categoricals
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print(f"  Features after encoding: {X.shape[1]}")

    # 2i. Stratified subsample — keeps training fast on CPU
    if TRAIN_SAMPLE and len(X) > TRAIN_SAMPLE:
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=TRAIN_SAMPLE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"  Subsampled to {TRAIN_SAMPLE:,} rows for training "
              f"(positive rate: {y_sub.mean():.3f})")
        X, y = X_sub, y_sub

    # 2j. Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. MODEL CONFIGS
# ─────────────────────────────────────────────
def build_configs():
    """
    LR / DT / RF / SVM: use class_weight='balanced' (no data duplication,
    low memory, fast).
    KNN: no class_weight support — use SMOTE on the small training sample.
    All tuned with RandomizedSearchCV (3-fold, n_iter=5).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    return [
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=1000,
                    solver="lbfgs", random_state=RANDOM_STATE
                )),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv,
        ),
        (
            "Decision Tree",
            Pipeline([
                ("clf", DecisionTreeClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE
                )),
            ]),
            {
                "clf__max_depth":        [5, 10, 20, None],
                "clf__min_samples_leaf": [1, 5, 10],
                "clf__criterion":        ["gini", "entropy"],
            },
            cv,
        ),
        (
            "Random Forest",
            Pipeline([
                ("clf", RandomForestClassifier(
                    class_weight="balanced", n_jobs=-1,
                    random_state=RANDOM_STATE
                )),
            ]),
            {
                "clf__n_estimators":     [50],
                "clf__max_depth":        [10, 20],
                "clf__min_samples_leaf": [1, 5],
            },
            cv,
        ),
        (
            "KNN",
            ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote",  SMOTE(random_state=RANDOM_STATE)),
                ("clf",    KNeighborsClassifier(n_jobs=-1)),
            ]),
            {
                "clf__n_neighbors": [5, 11, 21],
                "clf__weights":     ["uniform", "distance"],
            },
            cv,
        ),
    ]


# ─────────────────────────────────────────────
# 4. TRAIN WITH HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def train_models(configs, X_train, y_train):
    trained = {}

    # --- Tuned models (RandomizedSearchCV) ---
    for name, pipe, param_dist, cv in configs:
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=N_ITER,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        print(f"  Best params : {search.best_params_}")
        print(f"  Best CV F1  : {search.best_score_:.4f}")
        trained[name] = search.best_estimator_

    # --- SVM: single fit with fixed C=1.0 (no CV search — too slow otherwise) ---
    print(f"\n{'─'*50}")
    print(f"  Training: SVM  (fixed C=1.0, no grid search)")
    print(f"{'─'*50}")
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0, class_weight="balanced",
            max_iter=1000, dual=False, random_state=RANDOM_STATE
        )),
    ])
    svm_pipe.fit(X_train, y_train)
    trained["SVM"] = svm_pipe
    print(f"  Done.")

    return trained


# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
def evaluate_models(trained, X_test, y_test):
    summary = []

    for name, model in trained.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        y_pred = model.predict(X_test)
        acc         = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Macro F1    : {f1_macro:.4f}")
        print(f"  Weighted F1 : {f1_weighted:.4f}")
        print()
        print(classification_report(
            y_test, y_pred,
            target_names=["Not Dropout", "Dropout"],
            zero_division=0
        ))

        # ROC-AUC
        roc_auc = None
        try:
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                # LinearSVC — use decision_function for binary ROC-AUC
                scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, scores)
            print(f"  ROC-AUC     : {roc_auc:.4f}")
        except Exception:
            print("  ROC-AUC     : N/A")

        # Confusion Matrix
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Dropout", "Dropout"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        safe = name.replace(" ", "_")
        fig.savefig(os.path.join(OUTPUT_DIR, f"cm_{safe}.png"),
                    bbox_inches="tight", dpi=120)
        plt.close(fig)

        summary.append({
            "Model":       name,
            "Accuracy":    round(acc, 4),
            "Macro_F1":    round(f1_macro, 4),
            "Weighted_F1": round(f1_weighted, 4),
            "ROC_AUC":     round(roc_auc, 4) if roc_auc is not None else "N/A",
        })

    summary_df = (
        pd.DataFrame(summary)
        .sort_values("Macro_F1", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "classification_summary.csv"), index=False)

    print(f"\n{'='*50}")
    print("  FINAL SUMMARY  (sorted by Macro F1)")
    print(f"{'='*50}")
    print(summary_df.to_string(index=False))
    return summary_df


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
def plot_feature_importance(trained, X_train, top_n=20):
    model = trained.get("Random Forest")
    if model is None:
        return
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return

    importances   = clf.feature_importances_
    feature_names = X_train.columns.tolist()
    if len(importances) != len(feature_names):
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color="steelblue"
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("  Feature importance plot saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  STUDENT DROPOUT — CLASSIFICATION")
    print("=" * 50)

    print("\n[1] Loading data ...")
    df = load_data()

    print("\n[2] Preprocessing ...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("\n[3] Training (RandomizedSearchCV, 3-fold) ...")
    configs = build_configs()
    trained = train_models(configs, X_train, y_train)

    print("\n[4] Evaluating on held-out test set ...")
    evaluate_models(trained, X_test, y_test)

    print("\n[5] Feature importance ...")
    plot_feature_importance(trained, X_train)

    print(f"\n  Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
