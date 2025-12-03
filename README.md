# septic_shock_icu
# Heterogeneous Corticosteroid Effects in Septic-shock

This repository contains the core code used to:
1. Learn low-dimensional representations of septic-shock trajectories from longitudinal ICU data.
2. Derive progression subphenotypes via consensus clustering.
3. Train an early-window XGBoost classifier to predict subphenotypes.
4. (Optional, real-data only) Build target-trial-ready cohorts for corticosteroid strategies and estimate IPTW weights.

The public repo ships with **simulated demo data** only (`data_demo/`).  
No real patient-level data are included.

---

## 1. Repository structure

```text
septic_shock_peer_review/
├─ data_demo/                 # Small synthetic demo cohort
│  ├─ static_sim.csv          # Static baseline covariates (1 row per stay_id)
│  ├─ dynamic_sim.csv         # Longitudinal time-series (multiple rows per stay_id)
│  └─ outcome_sim.csv         # Outcomes for demo cohort
│
├─ results/                   # Output directory (created during runs)
│
├─ src/
│  ├─ representations/        # Representation learning (LSTM autoencoder)
│  │  ├─ build_tensors_1.py
│  │  ├─ datamodule_2.py
│  │  ├─ train_repr_3.py
│  │  └─ extract_embed_4.py
│  │
│  ├─ clusters/               # Consensus clustering and diagnostics
│  │  ├─ cluster_consensus.py
│  │  └─ cluster_consensus_eval.py
│  │
│  ├─ ett/                    # Target trial emulation utilities (real data only)
│  │  ├─ run_1.py
│  │  ├─ build_dataset_2.py
│  │  └─ balance_3.py
│  │
│  └─ predict/                # Subphenotype prediction
│     └─ xgboost_model.py
│
└─ README.md
