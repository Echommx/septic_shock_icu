#!/usr/bin/env python
"""
train_repr_3.py

Train the joint BiLSTM autoencoder that encodes longitudinal and static
features of septic-shock patients into a 32-dimensional representation.
"""
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from septic_shock_peer_review.src.representations.datamodule_2 import make_loaders

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)


class JointBiLSTMAE(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, hid: int = 64, emb_dim: int = 32):
        super().__init__()
        self.enc = nn.LSTM(seq_dim, hid, batch_first=True, bidirectional=True)
        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hid * 2),
        )
        self.proj = nn.Linear(hid * 4, emb_dim)
        self.dec = nn.LSTM(emb_dim, hid * 2, batch_first=True)
        self.dec_proj = nn.Linear(hid * 2, seq_dim)
        self.static_dec = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, static_dim),
        )

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor, x_static: torch.Tensor):
        B, T, _ = x_seq.shape
        seq_lat, _ = self.enc(x_seq)
        seq_lat = seq_lat[:, -1]
        stat_lat = self.static_enc(x_static)
        z = self.proj(torch.cat([seq_lat, stat_lat], dim=1))
        z_expanded = z.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.dec(z_expanded)
        dec_out = self.dec_proj(dec_out)
        static_rec = self.static_dec(z)
        seq_rec_loss = ((dec_out - x_seq) ** 2 * mask).mean()
        static_rec_loss = (static_rec - x_static).pow(2).mean()
        total_loss = seq_rec_loss + static_rec_loss
        return z, total_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    batch_size = 256

    loaders = make_loaders(batch=batch_size, num_workers=0)
    _, _, seq_dim = loaders["train"].dataset.x_seq.shape
    static_dim = loaders["train"].dataset.x_static.shape[1]

    model = JointBiLSTMAE(seq_dim=seq_dim, static_dim=static_dim).to(device)
    opt = Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loaders["train"]:
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            stat = batch["static"].to(device)
            _, loss = model(seq, mask, stat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * seq.size(0)
        train_loss = running / len(loaders["train"].dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)
                stat = batch["static"].to(device)
                _, loss = model(seq, mask, stat)
                running += loss.item() * seq.size(0)
        val_loss = running / len(loaders["val"].dataset)
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    model_dir = Path("septic_shock_peer_review/results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / "joint_ae.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Model saved to {ckpt.resolve()}")


if __name__ == "__main__":
    main()
