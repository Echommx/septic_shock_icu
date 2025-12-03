#!/usr/bin/env python
"""
extract_embed_4.py

Load the trained JointBiLSTMAE model and export the 32-dimensional
representations for all patients in the train, validation and test splits.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from septic_shock_peer_review.src.representations.train_repr_3 import JointBiLSTMAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="septic_shock_peer_review/results/data_proc",
        help="Directory containing *_tensor.pt files",
    )
    parser.add_argument(
        "--ckpt",
        default="septic_shock_peer_review/results/models/joint_ae.pt",
        help="Path to trained model checkpoint",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt = Path(args.ckpt)
    outfile = data_dir / "embed.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp = torch.load(data_dir / "train_tensor.pt")
    T, D = tmp["x_seq"].shape[1:3]
    S = tmp["x_static"].shape[1]

    model = JointBiLSTMAE(seq_dim=D, static_dim=S).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    embeds = []
    stay_ids = []

    for split in ("train", "val", "test"):
        ds = torch.load(data_dir / f"{split}_tensor.pt")
        x_seq = torch.from_numpy(ds["x_seq"]).float()
        x_seq = torch.nan_to_num(x_seq, nan=0.0)
        mask = torch.from_numpy(ds["mask"]).float()
        x_static = torch.from_numpy(ds["x_static"]).float()

        x_seq = x_seq.to(device)
        mask = mask.to(device)
        x_static = x_static.to(device)

        with torch.no_grad():
            z, _ = model(x_seq, mask, x_static)
        embeds.append(z.cpu().numpy())
        stay_ids.append(ds.get("stay_id"))

    embeds_arr = np.vstack(embeds).astype(np.float32)
    stay_ids_arr = np.concatenate(stay_ids)

    np.savez_compressed(outfile, embed=embeds_arr, stay_id=stay_ids_arr)
    print(f"Embeddings saved to {outfile.resolve()}")


if __name__ == "__main__":
    main()

