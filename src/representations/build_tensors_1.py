#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_tensors_1.py

Prepare static and dynamic tensors for representation learning on the demo data:
    - read static_sim.csv and dynamic_sim.csv
    - align rows by stay_id and pivot dynamic data to (N, T, D)
    - build static feature matrix with median/mode imputation,
      standardisation and one-hot encoding
    - treat extreme dynamic values as missing and create missingness masks
    - split into train/val/test (70/15/15) and save tensors to disk
"""
import argparse
import json
import random
import warnings
from pathlib import Path
import fnmatch
import re

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 2025
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_STATIC = "septic_shock_peer_review/data_demo/static_sim.csv"
DEFAULT_DYNAMIC = "septic_shock_peer_review/data_demo/dynamic_sim.csv"


def _expand_drop(columns, spec):
    selected = set()
    for token in spec or []:
        if not token:
            continue
        if token.startswith("re:"):
            try:
                pattern = re.compile(token[3:])
            except re.error:
                continue
            selected.update([c for c in columns if pattern.search(c)])
        elif any(ch in token for ch in "*?"):
            selected.update(fnmatch.filter(columns, token))
        elif token in columns:
            selected.add(token)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static_csv",
        default=DEFAULT_STATIC,
        help="Path to static csv (one row per stay_id)",
    )
    parser.add_argument(
        "--dynamic_csv",
        default=DEFAULT_DYNAMIC,
        help="Path to dynamic csv in long format with bucket_id",
    )
    parser.add_argument(
        "--out_dir",
        default="septic_shock_peer_review/results/data_proc",
        help="Output directory for tensors and transformers",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_col = "stay_id"
    time_col = "bucket_id"
    label_col = None

    static_df = pd.read_csv(args.static_csv)
    dynamic_df = pd.read_csv(args.dynamic_csv)

    if id_col not in static_df.columns:
        raise KeyError(f"{id_col} not found in static_df")
    if id_col not in dynamic_df.columns or time_col not in dynamic_df.columns:
        raise KeyError(f"{id_col} or {time_col} not found in dynamic_df")

    static_df = static_df.sort_values(id_col).reset_index(drop=True)
    dynamic_df = dynamic_df.sort_values([id_col, time_col]).reset_index(drop=True)

    pivot = dynamic_df.pivot(index=id_col, columns=time_col).sort_index()
    pivot = pivot.reindex(static_df[id_col])

    if pivot.shape[0] != static_df.shape[0]:
        raise RuntimeError("Row mismatch between static and dynamic tables")

    num_cols = [
        c
        for c in static_df.columns
        if c not in (id_col, label_col) and pd.api.types.is_numeric_dtype(static_df[c])
    ]
    cat_cols = [
        c
        for c in static_df.columns
        if c not in (id_col, label_col) and c not in num_cols
    ]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    ct = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    x_static = ct.fit_transform(static_df)
    if sp.issparse(x_static):
        x_static = x_static.toarray()
    x_static = x_static.astype(np.float32)

    joblib.dump(ct, out_dir / "static_ct.joblib")
    with open(out_dir / "static_cols.json", "w") as f:
        json.dump({"num": num_cols, "cat": cat_cols}, f, indent=2)

    T = len(pivot.columns.levels[1])
    seq_cols = [c for c in dynamic_df.columns if c not in (id_col, time_col)]
    D = len(seq_cols)

    x_seq = pivot.values.reshape(static_df.shape[0], T, D).astype(np.float32)
    x_seq[np.isinf(x_seq)] = np.nan

    low = np.nanpercentile(x_seq, 0.1, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(x_seq, 99.9, axis=(0, 1), keepdims=True)
    outlier_mask = (x_seq < low) | (x_seq > high)
    x_seq[outlier_mask] = np.nan

    mask = (~np.isnan(x_seq)).astype(np.float32)

    np.savez_compressed(out_dir / "seq_and_mask.npz", x_seq=x_seq, mask=mask)

    idx = np.arange(x_static.shape[0])
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, shuffle=True, random_state=SEED
    )
    val_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, shuffle=True, random_state=SEED
    )
    splits = {"train": tr_idx, "val": val_idx, "test": te_idx}

    with open(out_dir / "splits.json", "w") as fp:
        json.dump({k: v.tolist() for k, v in splits.items()}, fp, indent=2)

    for name, ix in splits.items():
        torch.save(
            {
                "x_static": x_static[ix],
                "x_seq": x_seq[ix],
                "mask": mask[ix],
                "stay_id": static_df[id_col].values[ix],
            },
            out_dir / f"{name}_tensor.pt",
        )


if __name__ == "__main__":
    main()
