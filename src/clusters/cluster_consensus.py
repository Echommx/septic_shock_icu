#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_consensus_1.py

Efficient consensus clustering on latent representations:
    - load embed.npz (embed, stay_id)
    - for each candidate k, build a consensus matrix by repeated subsampling
      and KMeans clustering
    - compute PAC and standard clustering metrics (silhouette, CH, DB)
    - save consensus matrices, cluster assignments and metrics to disk
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from tqdm import trange

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="septic_shock_peer_review/results/data_proc",
        help="Directory containing embed.npz",
    )
    ap.add_argument(
        "--out_dir",
        default="septic_shock_peer_review/results/data_clusters/consensus",
        help="Output directory for consensus results",
    )
    ap.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum number of clusters to consider",
    )
    ap.add_argument(
        "--k_max",
        type=int,
        default=9,
        help="Maximum number of clusters to consider",
    )
    ap.add_argument(
        "--boot",
        type=int,
        default=20, #The smaller the boot, the faster it runs,so Set it temporarily to 20
        help="Number of bootstrap resamples",
    )
    ap.add_argument(
        "--subsample_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to include in each resample",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Sample with replacement instead of without replacement",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed",
    )
    return ap.parse_args()


def build_consensus_matrix(Z, k, rng, boot, subsample_ratio, replace):
    N = Z.shape[0]
    samp = int(N * subsample_ratio)
    consensus_dict = {}

    for _ in trange(boot, desc=f"k={k}", leave=False):
        idx = rng.choice(N, samp, replace=replace)
        lab = KMeans(k, n_init="auto", random_state=rng.integers(1e9)).fit_predict(
            Z[idx]
        )
        same = lab[:, None] == lab[None, :]
        for i in range(len(idx)):
            for j in range(i, len(idx)):
                if same[i, j]:
                    key = (int(min(idx[i], idx[j])), int(max(idx[i], idx[j])))
                    consensus_dict[key] = consensus_dict.get(key, 0) + 1

    rows, cols, data = [], [], []
    for (i, j), count in consensus_dict.items():
        rows.append(i)
        cols.append(j)
        data.append(count / boot)

    C_sparse = coo_matrix(
        (data + data, (rows + cols, cols + rows)), shape=(N, N)
    ).tocsr()
    C_sparse.setdiag(1.0)
    return C_sparse.toarray().astype(np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    npz = np.load(Path(args.in_dir) / "embed.npz")
    Z = npz["embed"] if "embed" in npz else npz["Z"]
    ids = npz["stay_id"] if "stay_id" in npz else np.arange(len(Z))
    N, D = Z.shape
    print(
        f"[Consensus] N={N} dim={D} boot={args.boot} "
        f"replace={args.replace} subsample={args.subsample_ratio}"
    )

    metrics = {}
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n>>> k={k}")
        C = build_consensus_matrix(
            Z,
            k=k,
            rng=rng,
            boot=args.boot,
            subsample_ratio=args.subsample_ratio,
            replace=args.replace,
        )

        labels = KMeans(k, n_init="auto", random_state=args.seed).fit_predict(Z)

        triu_idx = np.triu_indices_from(C, k=1)
        pac = (
            ((C[triu_idx] > 0.1) & (C[triu_idx] < 0.9)).mean().item()
            if triu_idx[0].size > 0
            else 0.0
        )

        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)
        db = davies_bouldin_score(Z, labels)

        metrics[k] = dict(pac=pac, sil=sil, ch=ch, db=db)

        np.save(out_dir / f"consensus_k{k}.npy", C.astype(np.float16))
        pd.DataFrame({"stay_id": ids, "cluster": labels}).to_csv(
            out_dir / f"cluster_k{k}.csv", index=False
        )

        del C

    pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "k"}).to_csv(
        out_dir / "consensus_metrics.csv", index=False
    )

    with open(out_dir / "consensus_config.json", "w") as f:
        json.dump(
            {
                "k_min": args.k_min,
                "k_max": args.k_max,
                "boot": args.boot,
                "subsample_ratio": args.subsample_ratio,
                "replace": args.replace,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print("Consensus clustering completed. Results saved to", out_dir)


if __name__ == "__main__":
    main()

