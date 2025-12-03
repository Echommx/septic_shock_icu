#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgboost_model.py

Train a one-vs-rest XGBoost classifier to predict septic-shock progression
subphenotypes from early-window features in the combined MIMIC–eICU cohort.

Workflow:
    - load static_sim.csv, dynamic_sim.csv and cluster_k4.csv
    - slice dynamic data to an early time window around shock onset
    - build a sparse feature matrix combining numeric, dynamic and one-hot
      encoded categorical variables (NaNs preserved for XGBoost)
    - perform stratified 10-fold cross-validation with OneVsRestClassifier
    - compute ROC curves (per class, micro, macro) with bootstrap AUC
    - fit on the full dataset and save the trained model and preprocessor
"""
import gc
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, label_binarize

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")

RNG = 2024
np.random.seed(RNG)

RAW_DIR = Path("septic_shock_peer_review/data_demo")
CL_DIR = Path("septic_shock_peer_review/results/data_clusters/consensus")
OUT_DIR = Path("septic_shock_peer_review/results/results/predictor/xgboost")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CSV = RAW_DIR / "static_sim.csv"
DYN_CSV = RAW_DIR / "dynamic_sim.csv"
LABEL_CSV = CL_DIR / "cluster_k4.csv"


def bucket_slice(df_dyn: pd.DataFrame, lo: int = -1, hi: int = 4) -> pd.DataFrame:
    return df_dyn[df_dyn.bucket_id.between(lo, hi)]


def build_X(static_df: pd.DataFrame, dyn_df: pd.DataFrame):
    id_col = "stay_id"
    cat_cols = static_df.select_dtypes("object").columns.tolist()
    num_cols = [c for c in static_df.columns if c not in cat_cols + [id_col]]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_mat = ohe.fit_transform(static_df[cat_cols].fillna("missing"))
    cat_names = ohe.get_feature_names_out()

    num_mat = static_df[num_cols].to_numpy(np.float32)
    num_names = num_cols

    dyn_wide = dyn_df.pivot(index="stay_id", columns="bucket_id")
    dyn_wide = dyn_wide.reindex(static_df[id_col])
    dyn_data = dyn_wide.values
    dyn_mat = np.where(pd.isnull(dyn_data), np.nan, dyn_data).astype(np.float32)
    dyn_names = [f"{col}_{bucket}" for col, bucket in dyn_wide.columns]

    X_dense = np.hstack([num_mat, dyn_mat])
    X_sparse = sp.hstack([sp.csr_matrix(X_dense), cat_mat], format="csr")
    feat_names = num_names + dyn_names + cat_names.tolist()
    return X_sparse, feat_names, ohe, num_cols, dyn_wide.columns.tolist()


def dense(a):
    return a.toarray() if sp.issparse(a) else a


def auc_ci(y_true, y_score, n_boot: int = 1000):
    rng = np.random.default_rng(RNG)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
            aucs.append(auc(fpr, tpr))
        except ValueError:
            continue
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def main():
    st = pd.read_csv(STATIC_CSV)
    dyn = bucket_slice(pd.read_csv(DYN_CSV))
    lab = pd.read_csv(LABEL_CSV)

    common = sorted(set(st.stay_id) & set(dyn.stay_id) & set(lab.stay_id))
    st = st[st.stay_id.isin(common)].sort_values("stay_id").reset_index(drop=True)
    dyn = dyn[dyn.stay_id.isin(common)]
    lab = lab[lab.stay_id.isin(common)].sort_values("stay_id").reset_index(drop=True)

    X, feat_names, ohe, num_cols, dyn_cols = build_X(st, dyn)
    y = lab.cluster.to_numpy().astype(int)
    Y_bin = label_binarize(y, classes=[0, 1, 2, 3])

    print(f"Feature matrix: {X.shape}  (features = {len(feat_names)})")

    base_model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.08,
        eval_metric="auc",
        objective="binary:logistic",
        tree_method="hist",
        random_state=RNG,
    )

    skf = StratifiedKFold(10, shuffle=True, random_state=RNG)

    print("\n=== XGBoost (one-vs-rest) ===")
    clf = OneVsRestClassifier(base_model, n_jobs=1)
    y_prob = cross_val_predict(
        clf,
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=1,
        verbose=0,
    )

    plt.figure(figsize=(6, 5))
    colors = sns.color_palette("deep", 6)

    auc_per_class = []
    for k in range(4):
        fpr, tpr, _ = roc_curve(Y_bin[:, k], y_prob[:, k])
        m_auc, std_auc = auc_ci(Y_bin[:, k], y_prob[:, k])
        auc_per_class.append(m_auc)
        plt.plot(
            fpr,
            tpr,
            color=colors[k],
            lw=2,
            label=f"S{k} vs rest (AUC = {m_auc:.2f} ± {std_auc:.02f})",
        )

    fpr_micro, tpr_micro, _ = roc_curve(Y_bin.ravel(), y_prob.ravel())
    auc_micro, std_micro = auc_ci(Y_bin.ravel(), y_prob.ravel())

    fpr_macro = np.unique(
        np.concatenate([roc_curve(Y_bin[:, k], y_prob[:, k])[0] for k in range(4)])
    )
    tpr_macro = np.zeros_like(fpr_macro)
    for k in range(4):
        f_k, t_k, _ = roc_curve(Y_bin[:, k], y_prob[:, k])
        tpr_macro += np.interp(fpr_macro, f_k, t_k)
    tpr_macro /= 4
    auc_macro = float(np.mean(auc_per_class))
    std_macro = float(np.std(auc_per_class, ddof=1))

    plt.plot(
        fpr_micro,
        tpr_micro,
        color="grey",
        lw=2,
        label=f"Micro-avg (AUC = {auc_micro:.2f} ± {std_micro:.02f})",
    )
    plt.plot(
        fpr_macro,
        tpr_macro,
        color="black",
        ls="--",
        lw=2,
        label=f"Macro-avg (AUC = {auc_macro:.2f} ± {std_macro:.02f})",
    )

    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC curve - XGBoost (MIMIC–eICU, 10-fold CV)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_all.png", dpi=300)
    plt.close()

    print("Computing SHAP values on full data …")
    clf.fit(X, y)

    for k in range(4):
        print(f"  Subphenotype S{k} vs rest")
        idx_k = np.where(y == k)[0]
        if len(idx_k) > 500:
            rng = np.random.default_rng(RNG)
            idx_k = rng.choice(idx_k, 500, replace=False)
        X_vis = dense(X[idx_k])

        try:
            estimator = clf.estimators_[k]
            explainer = shap.TreeExplainer(estimator, model_output="raw")
            sv = explainer.shap_values(X_vis, check_additivity=False)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                sv,
                X_vis,
                feature_names=feat_names,
                plot_type="violin",
                max_display=15,
                show=False,
            )
            plt.title(f"XGBoost: Rest vs S{k}", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                OUT_DIR / f"S{k}_shap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"    SHAP error for S{k}: {e}")
        finally:
            del explainer, sv, X_vis
            gc.collect()

    print("\nSaving model and preprocessor …")
    import joblib
    import os

    os.makedirs("septic_shock_peer_review/results/models", exist_ok=True)
    joblib.dump(clf, "septic_shock_peer_review/results/models/xgb_model.pkl")
    preprocessor = {
        "ohe": ohe,
        "num_cols": num_cols,
        "dyn_cols": [tuple(c) for c in dyn_cols],
    }
    joblib.dump(preprocessor, "septic_shock_peer_review/results/models/xgb_preproc.joblib")

    print("Saved model → septic_shock_peer_review/results/models/xgb_model.pkl")
    print("Saved preprocessor → septic_shock_peer_review/results/models/xgb_preproc.joblib")
    print("\nAll figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
