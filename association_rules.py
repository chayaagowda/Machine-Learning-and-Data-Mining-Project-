"""
Association Rule Mining — Dropout Pattern Discovery
ITMD 522 | Joshua Godwin

Implements:
  - Equal-frequency discretization (Low / Medium / High) for continuous features
  - Apriori (mlxtend) and FP-Growth (mlxtend)
  - Rule filtering: min support 0.05, min confidence 0.65
  - Top 20 rules by Lift  (overall + dropout-as-consequent)
  - Apriori vs FP-Growth consistency check (frequent itemset count + runtime)

Outputs (under results/association_rules/):
  - frequent_itemsets_apriori.csv
  - frequent_itemsets_fpgrowth.csv
  - top20_rules_by_lift.csv
  - top20_dropout_rules.csv
  - algorithm_comparison.csv
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError as e:
    raise SystemExit(
        "mlxtend is required.  Install with:  "
        "pip install mlxtend --break-system-packages"
    ) from e

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "association_rules")
TARGET_COL      = "abandono_hash"
RANDOM_STATE    = 42
ARM_SAMPLE      = 50_000   # transactional matrices are dense — keep this manageable
MIN_SUPPORT          = 0.05   # main pass — proposal-spec
MIN_CONFIDENCE       = 0.65   # main pass — proposal-spec
DROPOUT_MIN_SUPPORT  = 0.01   # secondary pass — relaxed so dropout itemsets surface
                              # (dropout class is only ~6.4% of records — global baseline)
DROPOUT_MIN_CONF     = 0.12   # 2x the baseline dropout rate -> all surviving rules have Lift > ~2
TOP_N           = 20
N_BINS          = 3        # Low / Medium / High

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Continuous features to discretize. Names AFTER preprocessing.
CONTINUOUS_FEATURES = [
    "nota10_hash",                     # GPA-like grade
    "rendimiento_total",               # academic progression rate
    "cred_sup_total",                  # credits passed
    "cred_mat_total",                  # credits attempted
    "total_pft_events",                # LMS events (annual total)
    "total_pft_visits",                # LMS visits
    "total_pft_assignment_submissions",
    "total_n_wifi_days",               # campus presence
    "total_resource_events",           # LMS resource accesses
]

# Categorical features to keep as-is (after re-naming to readable labels).
CATEGORICAL_KEEP = [
    "dedicacion",        # TC = full-time, TP = part-time
    "tipo_ingreso",      # admission pathway
    "campus_hash",       # campus
    "estudios_p_hash",   # parent's studies (T/F)
    "estudios_m_hash",
    "desplazado_hash",
]

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
# 1. LOAD + LIGHT PREPROCESS
# ─────────────────────────────────────────────
def load_and_prepare():
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

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Encode dropout
    df = df.dropna(subset=[TARGET_COL]).copy()
    df["Dropout"] = np.where(df[TARGET_COL] == "A", "Yes", "No")
    df.drop(columns=[TARGET_COL], inplace=True)

    # Drop ID / leakage cols
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Comma-as-decimal fix
    for col in COMMA_DECIMAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # Aggregate monthly LMS / Wi-Fi columns into yearly totals
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

    print(f"  After aggregation: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Subsample (stratified on Dropout)
    if ARM_SAMPLE and len(df) > ARM_SAMPLE:
        df = (
            df.groupby("Dropout", group_keys=False)
              .apply(lambda g: g.sample(
                  n=int(round(ARM_SAMPLE * len(g) / len(df))),
                  random_state=RANDOM_STATE
              ))
              .reset_index(drop=True)
        )
        print(f"  Stratified subsample: {len(df):,} rows "
              f"(positive rate: {(df['Dropout']=='Yes').mean():.3f})")

    return df


# ─────────────────────────────────────────────
# 2. DISCRETIZE
# ─────────────────────────────────────────────
def discretize(df):
    """
    Build a transactional DataFrame:
      one column per kept feature, values are short readable labels
      e.g. 'nota10_hash=Low', 'dedicacion=TC', 'Dropout=Yes'
    """
    out = pd.DataFrame(index=df.index)

    # 2a. Discretize continuous features into Low / Medium / High (equal-frequency)
    for col in CONTINUOUS_FEATURES:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # qcut requires no NaNs, and unique enough quantiles. Drop NaNs first;
        # rows with NaNs simply won't carry that item.
        if s.notna().sum() == 0:
            continue
        try:
            binned = pd.qcut(
                s, q=N_BINS, labels=["Low", "Medium", "High"], duplicates="drop"
            )
        except ValueError:
            # qcut can fail on heavily skewed columns where >q duplicates exist
            try:
                binned = pd.cut(
                    s, bins=N_BINS, labels=["Low", "Medium", "High"]
                )
            except Exception:
                continue
        out[col] = binned.astype(str).where(binned.notna(), other=np.nan)
        out[col] = out[col].apply(lambda v: f"{col}={v}" if isinstance(v, str) and v != "nan" else np.nan)

    # 2b. Keep selected categorical columns as-is (with prefix)
    for col in CATEGORICAL_KEEP:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        out[col] = s.apply(lambda v: f"{col}={v}" if v not in ("nan", "Unknown", "") else np.nan)

    # 2c. Target
    out["Dropout"] = "Dropout=" + df["Dropout"].astype(str)

    # Each row -> list of items (drop NaNs)
    transactions = out.apply(lambda r: [v for v in r.values if isinstance(v, str)], axis=1).tolist()
    print(f"  Transactions: {len(transactions):,} | "
          f"avg items/row: {np.mean([len(t) for t in transactions]):.1f}")
    return transactions


# ─────────────────────────────────────────────
# 3. BUILD ONE-HOT TRANSACTION MATRIX
# ─────────────────────────────────────────────
def build_transaction_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    # mlxtend >= 0.20 emits a DeprecationWarning if the input isn't bool.
    tdf = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)
    print(f"  One-hot transaction matrix: {tdf.shape}")
    return tdf


# ─────────────────────────────────────────────
# 4. RUN APRIORI + FP-GROWTH
# ─────────────────────────────────────────────
def run_frequent(tdf):
    print(f"\n  Running Apriori (min_support={MIN_SUPPORT}) ...")
    t0 = time.time()
    fi_apriori = apriori(tdf, min_support=MIN_SUPPORT, use_colnames=True)
    apriori_time = time.time() - t0
    print(f"    Apriori   -> {len(fi_apriori):,} itemsets in {apriori_time:.2f}s")

    print(f"\n  Running FP-Growth (min_support={MIN_SUPPORT}) ...")
    t0 = time.time()
    fi_fpg = fpgrowth(tdf, min_support=MIN_SUPPORT, use_colnames=True)
    fpg_time = time.time() - t0
    print(f"    FP-Growth -> {len(fi_fpg):,} itemsets in {fpg_time:.2f}s")

    fi_apriori.to_csv(os.path.join(OUTPUT_DIR, "frequent_itemsets_apriori.csv"), index=False)
    fi_fpg.to_csv(os.path.join(OUTPUT_DIR, "frequent_itemsets_fpgrowth.csv"), index=False)

    cmp_df = pd.DataFrame([
        {"Algorithm": "Apriori",  "n_itemsets": len(fi_apriori), "runtime_sec": round(apriori_time, 3)},
        {"Algorithm": "FP-Growth", "n_itemsets": len(fi_fpg),     "runtime_sec": round(fpg_time, 3)},
    ])
    cmp_df.to_csv(os.path.join(OUTPUT_DIR, "algorithm_comparison.csv"), index=False)
    return fi_apriori, fi_fpg, cmp_df


# ─────────────────────────────────────────────
# 5. RULES
# ─────────────────────────────────────────────
def rules_from(itemsets, tdf, min_conf=MIN_CONFIDENCE):
    """Generate rules from frequent itemsets, then filter by min confidence."""
    if len(itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence",
                              min_threshold=min_conf)
    # readable string forms
    rules["antecedents_str"]  = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents_str"]  = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules


def top_rules(rules, n=TOP_N):
    cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    return (
        rules.sort_values("lift", ascending=False)
             .head(n)[cols]
             .reset_index(drop=True)
             .round(4)
    )


def dropout_rules(rules, n=TOP_N):
    """Rules where Dropout=Yes is anywhere in the consequent set."""
    if rules.empty:
        return rules
    mask = rules["consequents"].apply(lambda s: "Dropout=Yes" in s)
    return top_rules(rules[mask], n=n)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ASSOCIATION RULE MINING — DROPOUT PATTERNS")
    print("=" * 60)

    print("\n[1] Loading + light preprocessing ...")
    df = load_and_prepare()

    print("\n[2] Discretizing into Low / Medium / High items ...")
    transactions = discretize(df)

    print("\n[3] Building transaction matrix ...")
    tdf = build_transaction_df(transactions)

    print("\n[4] Mining frequent itemsets (Apriori + FP-Growth) ...")
    fi_apriori, fi_fpg, cmp_df = run_frequent(tdf)
    print("\n  ALGORITHM COMPARISON")
    print(cmp_df.to_string(index=False))

    print(f"\n[5] Generating rules (min_confidence={MIN_CONFIDENCE}) ...")
    # Use FP-Growth itemsets for rule generation (same itemsets as Apriori,
    # but faster on this dataset). Apriori itemsets are saved separately.
    rules = rules_from(fi_fpg, tdf, MIN_CONFIDENCE)
    print(f"  Total rules above thresholds: {len(rules):,}")

    print(f"\n[6] Top {TOP_N} rules by Lift (any consequent) ...")
    top = top_rules(rules, TOP_N)
    top.to_csv(os.path.join(OUTPUT_DIR, "top20_rules_by_lift.csv"), index=False)
    print(top.to_string(index=False))

    # ─── Secondary pass: relaxed support, lift-based filter for Dropout=Yes ───
    # The dropout class is ~6% of records, so the proposal-spec support 0.05 +
    # confidence 0.65 is structurally infeasible for Dropout=Yes-as-consequent
    # rules. We rerun at DROPOUT_MIN_SUPPORT and filter by Lift > 1, which is
    # the proposal's actual recommended ranking metric anyway.
    print(f"\n[7] Secondary pass for dropout rules "
          f"(min_support={DROPOUT_MIN_SUPPORT}, sort by Lift > 1) ...")
    fi_drop = fpgrowth(tdf, min_support=DROPOUT_MIN_SUPPORT, use_colnames=True)
    print(f"  Frequent itemsets at relaxed support: {len(fi_drop):,}")

    # How many of those itemsets actually contain Dropout=Yes?
    contains_dy = fi_drop["itemsets"].apply(lambda s: "Dropout=Yes" in s)
    print(f"  Itemsets containing Dropout=Yes: {int(contains_dy.sum()):,}")

    # Generate rules with NO confidence floor — sort/filter by Lift instead.
    # association_rules supports metric='lift' with min_threshold=1.0
    if len(fi_drop) > 0:
        all_rules = association_rules(fi_drop, metric="lift", min_threshold=1.0)
        all_rules["antecedents_str"] = all_rules["antecedents"].apply(
            lambda s: ", ".join(sorted(s))
        )
        all_rules["consequents_str"] = all_rules["consequents"].apply(
            lambda s: ", ".join(sorted(s))
        )
    else:
        all_rules = pd.DataFrame()

    print(f"  Total rules with Lift > 1: {len(all_rules):,}")

    # Restrict to rules where Dropout=Yes is the SOLE consequent — that's the
    # actionable form ("if X, then dropout").
    if not all_rules.empty:
        mask = all_rules["consequents"].apply(
            lambda s: s == frozenset({"Dropout=Yes"})
        )
        drop_rules = all_rules[mask].copy()
        print(f"  Rules with Dropout=Yes as sole consequent: {len(drop_rules):,}")
    else:
        drop_rules = pd.DataFrame()

    if not drop_rules.empty:
        cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        drop_top = (
            drop_rules.sort_values("lift", ascending=False)
                      .head(TOP_N)[cols]
                      .reset_index(drop=True)
                      .round(4)
        )
    else:
        drop_top = pd.DataFrame(columns=["antecedents_str", "consequents_str",
                                         "support", "confidence", "lift"])

    drop_top.to_csv(os.path.join(OUTPUT_DIR, "top20_dropout_rules.csv"), index=False)
    if drop_top.empty:
        print("  No rules surfaced for Dropout=Yes — the dataset's dropout class "
              "may be too rare/heterogeneous for this thresholding.")
    else:
        print(f"\n  Top {len(drop_top)} dropout rules (sorted by Lift):")
        print(drop_top.to_string(index=False))

    print(f"\n  Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
