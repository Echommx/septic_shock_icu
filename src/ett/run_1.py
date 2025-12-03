#!/usr/bin/env python
# coding: utf-8
"""
run_1.py

Build a target-trial-ready dataset for the cortico cohort using
pre-computed consensus clusters as subphenotype labels:
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = Path("cortico_ett/mezxett/mg200/data_raw")
PROC_DIR = Path("cortico_ett/mezxett/mg200/data_proc/cortico")
RESULT_DIR = Path("cortico_ett/mezxett/mg200/results/cortico")
CLUSTER_DIR = Path("cortico_ett/mezxett/mg200/data_raw")

PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CSV = RAW_DIR / "cortico_mezxett_variable.csv"
STRATEGY_CSV = RAW_DIR / "cortico_mezxett_strategy.csv"
OUTCOME_CSV = RAW_DIR / "cortico_mezxett_outcome.csv"
CONSENSUS_CSV = CLUSTER_DIR / "cluster_mezxett_k4.csv"

EARLY_CSV = PROC_DIR / "early_static.csv"
LABEL_CSV = PROC_DIR / "labels_pred.csv"
TT_CSV = PROC_DIR / "tt_dataset_pred.csv"
HIST_PNG = RESULT_DIR / "label_hist.png"


def step1_make_static():
    df = pd.read_csv(STATIC_CSV)
    if "stay_id" not in df.columns:
        raise KeyError("Static table must contain column 'stay_id'.")

    if df["stay_id"].duplicated().any():
        dups = df[df["stay_id"].duplicated()]["stay_id"].tolist()[:5]
        raise ValueError(
            f"Duplicate stay_id values found in static table (examples: {dups} ...). "
            "Please de-duplicate before proceeding."
        )

    df.to_csv(EARLY_CSV, index=False)
    print("Static baseline written to", EARLY_CSV)


def step2_assign_label_from_consensus():
    lab = pd.read_csv(CONSENSUS_CSV)

    norm = {c.strip().lower(): c for c in lab.columns}
    if "stay_id" not in norm or "cluster" not in norm:
        raise KeyError(
            "Consensus file must contain columns 'stay_id' and 'cluster' "
            "(case and surrounding spaces are ignored)."
        )

    id_col = norm["stay_id"]
    cluster_col = norm["cluster"]

    lab = (
        lab[[id_col, cluster_col]]
        .rename(columns={id_col: "stay_id", cluster_col: "cluster"})
        .drop_duplicates(subset=["stay_id"])
        .sort_values("stay_id")
    )

    lab["subphenotype"] = lab["cluster"]

    lab[["stay_id", "cluster", "subphenotype"]].to_csv(LABEL_CSV, index=False)
    print("Labels (from consensus) written to", LABEL_CSV)

    plt.figure(figsize=(5, 3))
    (
        lab["cluster"]
        .value_counts()
        .sort_index()
        .plot(kind="bar")
    )
    plt.title("Cluster Distribution (consensus k=4)")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=200)
    plt.close()
    print("Cluster histogram saved to", HIST_PNG)


def step3_merge_trial():
    strategy = pd.read_csv(STRATEGY_CSV)
    outcome = pd.read_csv(OUTCOME_CSV)
    labels = pd.read_csv(LABEL_CSV)

    for name, df in [("strategy", strategy), ("outcome", outcome), ("labels", labels)]:
        if "stay_id" not in df.columns:
            raise KeyError(f"{name} table must contain column 'stay_id'.")

    trial_df = (
        strategy.merge(
            labels[["stay_id", "cluster", "subphenotype"]],
            on="stay_id",
            how="inner",
        )
        .merge(outcome, on="stay_id", how="inner")
    )

    trial_df.to_csv(TT_CSV, index=False)
    print("Trial-ready dataset written to", TT_CSV)


if __name__ == "__main__":
    step1_make_static()
    step2_assign_label_from_consensus()
    step3_merge_trial()
    print("Done. See processed data in", PROC_DIR, "and results in", RESULT_DIR)
