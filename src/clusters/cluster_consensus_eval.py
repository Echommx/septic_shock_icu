#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_consensus_eval.py

Evaluate consensus clustering results on latent representations:
    - align cluster labels across k values for consistent tracking
    - compute CDF, AUC, PAC and within-cluster similarity
    - generate IC histograms, IC bar plots and consensus heatmaps
    - compute standard external cluster validity indices
"""
import argparse
import gc
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        default="septic_shock_peer_review/results/data_clusters/consensus",
        help="Directory containing consensus_k*.npy and cluster_k*.csv",
    )
    parser.add_argument(
        "--cons_dir",
        default="septic_shock_peer_review/results/data_proc",
        help="Unused compatibility argument",
    )
    parser.add_argument(
        "--out_dir",
        default="septic_shock_peer_review/results/eval/consensus",
        help="Output directory for evaluation figures and tables",
    )
    parser.add_argument(
        "--embed_dir",
        default="septic_shock_peer_review/results/data_proc",
        help="Directory containing embed.npz",
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum k to evaluate",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=9,
        help="Maximum k to evaluate",
    )
    parser.add_argument(
        "--heat_max",
        type=int,
        default=2000,
        help="Maximum number of items to show in heatmaps",
    )
    parser.add_argument(
        "--cdf_points",
        type=int,
        default=1000,
        help="Number of points to subsample CDF for plotting",
    )
    return parser.parse_args()


def align_labels(prev, curr, next_new_id):
    up = prev.max() + 1
    uc = curr.max() + 1
    overlap = np.zeros((up, uc), dtype=int)
    for i in range(up):
        for j in range(uc):
            overlap[i, j] = np.sum((prev == i) & (curr == j))
    row, col = linear_sum_assignment(-overlap)
    mapping = {}
    for r, c in zip(row, col):
        if overlap[r, c] > 0:
            mapping[c] = r
    new_curr = curr.copy()
    for j in range(uc):
        if j in mapping:
            new_curr[curr == j] = mapping[j]
        else:
            new_curr[curr == j] = next_new_id
            next_new_id += 1
    return new_curr, next_new_id


def build_cmap(n_colors):
    base = (
        list(plt.cm.get_cmap("tab20").colors)
        + list(plt.cm.get_cmap("tab20b").colors)
        + list(plt.cm.get_cmap("tab20c").colors)
    )
    repeats = int(np.ceil(n_colors / len(base)))
    colors = base * repeats
    return ListedColormap(colors[:n_colors])


def main():
    args = parse_args()

    in_dir = Path(args.in_dir)
    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(embed_dir / "embed.npz")
    Z = npz["embed"] if "embed" in npz else npz["Z"]
    N, _ = Z.shape

    auc_dict = {}
    pac_dict = {}
    mean_sim_dict = {}
    cdf_sampled = {}
    cluster_metrics = []
    label_matrix_aligned = []
    next_global_id = 0

    for ik, k in enumerate(tqdm(range(args.k_min, args.k_max + 1), desc="Eval k")):
        C = np.load(in_dir / f"consensus_k{k}.npy").astype(np.float32)
        lbl = pd.read_csv(in_dir / f"cluster_k{k}.csv")["cluster"].values

        if ik == 0:
            new_lbl = lbl.copy()
            next_global_id = int(new_lbl.max()) + 1
        else:
            prev_lbl = label_matrix_aligned[-1]
            new_lbl, next_global_id = align_labels(prev_lbl, lbl, next_global_id)
        label_matrix_aligned.append(new_lbl)

        tri = C[np.triu_indices_from(C, 1)]
        tri_s = np.sort(tri)[:: max(1, len(tri) // args.cdf_points)]
        cdf = np.arange(1, len(tri_s) + 1) / len(tri_s)
        auc = np.trapz(cdf, tri_s)
        auc_dict[k] = float(auc)
        cdf_sampled[k] = (tri_s, cdf)

        pac = ((tri > 0.10) & (tri < 0.90)).mean()
        pac_dict[k] = float(pac)

        sim_k = []
        for cid in np.unique(new_lbl):
            m = new_lbl == cid
            if m.sum() <= 1:
                sim_k.append(1.0)
            else:
                sub = C[m][:, m]
                np.fill_diagonal(sub, 0)
                sim_k.append(sub.sum() / (m.sum() * (m.sum() - 1)))
        mean_sim_dict[k] = sim_k

        ic = np.zeros(N)
        for i in range(N):
            peer = (new_lbl == new_lbl[i]) & (np.arange(N) != i)
            if peer.any():
                ic[i] = C[i, peer].mean()
            else:
                ic[i] = 1.0

        cmap = build_cmap(next_global_id)

        bins = np.linspace(0, 1, 21)
        plt.figure(figsize=(8, 5))
        plt.hist(ic, bins=bins, density=True, alpha=0.8)
        plt.axvline(ic.mean(), ls="--", color="red", label=f"Mean={ic.mean():.3f}")
        plt.xlim(0, 1)
        plt.title(f"IC Distribution (k={k})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"ic_k{k}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        for cid in np.unique(new_lbl):
            plt.hist(
                ic[new_lbl == cid],
                bins=bins,
                density=True,
                alpha=0.6,
                color=cmap(cid),
                label=f"C{cid}",
            )
        plt.xlim(0, 1)
        plt.title(f"IC Distribution by Cluster (k={k})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"ic_k{k}_by_cluster.png", dpi=300)
        plt.close()

        order_ic = np.argsort(ic)[::-1]
        plt.figure(figsize=(10, 1.8))
        plt.bar(
            np.arange(N),
            ic[order_ic],
            color=[cmap(c) for c in new_lbl[order_ic]],
            width=1.0,
        )
        plt.ylim(0, 1.05)
        plt.xticks([])
        plt.yticks([0, 0.5, 1])
        plt.title(f"item-consensus k={k}", fontsize=10, pad=6)
        ax = plt.gca()
        for spine in ["left", "right", "top"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / f"ic_bar_k{k}.png", dpi=300)
        plt.close()

        if N <= args.heat_max:
            show = np.arange(N)
        else:
            show = np.random.default_rng(2025).choice(
                N, args.heat_max, replace=False
            )
        order_sub = np.argsort(new_lbl[show])
        subC = C[show][:, show][order_sub][:, order_sub]
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            subC,
            cmap="viridis",
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            vmin=0,
            vmax=1,
        )
        plt.title(f"Consensus Matrix (k={k}, n={len(show)})")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_k{k}.png", dpi=300)
        plt.close()

        sil = silhouette_score(Z, new_lbl)
        ch = calinski_harabasz_score(Z, new_lbl)
        db = davies_bouldin_score(Z, new_lbl)
        cnt = np.bincount(new_lbl)
        cluster_metrics.append(
            dict(
                k=k,
                Silhouette=float(sil),
                CH=float(ch),
                DB=float(db),
                MinSize=int(cnt.min()),
                MaxSize=int(cnt.max()),
            )
        )

        del C, subC
        gc.collect()

    k_vals = list(range(args.k_min, args.k_max + 1))
    cmap_k = build_cmap(len(k_vals) + 5)

    plt.figure(figsize=(9, 6))
    for i, k in enumerate(k_vals):
        xs, cd = cdf_sampled[k]
        plt.step(xs, cd, where="post", color=cmap_k(i), label=f"k={k}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Similarity")
    plt.ylabel("Cumulative probability")
    plt.title("Cumulative distribution function of similarity")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_allk.png", dpi=300)
    plt.close()

    delta_auc = [auc_dict[k_vals[0]]]
    for k in k_vals[1:]:
        delta_auc.append(auc_dict[k] - auc_dict[k - 1])
    plt.figure(figsize=(7, 4))
    plt.plot(k_vals, delta_auc, "o-", lw=2, ms=6)
    plt.xlabel("k")
    plt.ylabel("Relative change in area under CDF curve")
    plt.title("Relative change in AUC")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "delta_auc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    pos = 0
    xt = []
    xt_lbl = []
    for i, k in enumerate(k_vals):
        vals = mean_sim_dict[k]
        xs = np.arange(len(vals)) + pos
        plt.bar(xs, vals, color=cmap_k(i), width=0.8, alpha=0.9)
        xt.append(xs.mean())
        xt_lbl.append(f"k={k}")
        pos += len(vals) + 1
    plt.xticks(xt, xt_lbl)
    plt.ylim(0, 1)
    plt.ylabel("Average similarity")
    plt.title("Average similarity of subphenotypes")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_sim_detail.png", dpi=300)
    plt.close()

    pd.DataFrame({"k": k_vals, "PAC": [pac_dict[k] for k in k_vals]}).to_csv(
        out_dir / "pac_allk.csv", index=False
    )
    plt.figure(figsize=(7, 4))
    plt.plot(k_vals, [pac_dict[k] for k in k_vals], "s-", lw=2, ms=6)
    plt.xlabel("k")
    plt.ylabel("PAC")
    plt.title("Proportion of ambiguous clustering")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "pac.png", dpi=300)
    plt.close()

    mat = np.stack(label_matrix_aligned, axis=1)
    order_items = np.lexsort(mat[:, ::-1].T)
    mat = mat[order_items].T

    width = min(16, max(6, N * 0.009))
    height = max(2, len(k_vals) * 0.5)
    plt.figure(figsize=(width, height))
    cmap_all = build_cmap(int(mat.max()) + 1)
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap_all)
    plt.xticks([])
    plt.xlabel("Items (sorted by consensus)")
    plt.yticks(range(len(k_vals)), [f"k={k}" for k in k_vals])
    for y in np.arange(-0.5, len(k_vals), 1):
        plt.hlines(y, -0.5, N - 0.5, colors="white", lw=1)
    plt.title("Tracking plot", pad=8)
    plt.tight_layout()
    plt.savefig(out_dir / "tracking_plot.png", dpi=300)
    plt.close()

    pd.DataFrame(cluster_metrics).to_csv(
        out_dir / "cluster_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()


