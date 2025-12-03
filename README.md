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

**Tested with:**

Python ≥ 3.10

Recommended packages:

numpy

pandas

scipy

scikit-learn

matplotlib

seaborn

torch (PyTorch, for LSTM autoencoder)

xgboost

(optional, for survival analysis in full project) lifelines, plus R packages prodlim, riskRegression, survival, timereg.

Example conda environment:

conda create -n septic python=3.10
conda activate septic

pip install numpy pandas scipy scikit-learn matplotlib seaborn torch xgboost lifelines

3. End-to-end demo pipeline (using data_demo/)

All commands below assume you are in the project root septic_shock_peer_review/.

3.1. Step 1 – Build tensors for representation learning

From raw demo CSVs → 3D tensors (N, T, D) for time-series + static features.

python src/representations/build_tensors_1.py


This script:

Reads data_demo/static_sim.csv, data_demo/dynamic_sim.csv, data_demo/outcome_sim.csv.

Aligns static and dynamic features by stay_id.

Applies preprocessing consistent with the manuscript:

static: median imputation + scaling / one-hot encoding,

dynamic: extreme values (<0.1 / >99.9 percentiles) set to missing + missingness masks.

Randomly splits cohort into train/val/test (70/15/15).

Saves tensors and index splits under results/representations/ (or similar path defined in the script).

3.2. Step 2 – Train LSTM autoencoder for representation learning
python src/representations/train_repr_3.py


This script:

Uses datamodule_2.py to create PyTorch DataLoaders from the pre-built tensors.

Trains a joint BiLSTM–AE + MLP model to reconstruct:

longitudinal sequences with a missingness-aware reconstruction loss,

static covariates in parallel.

Projects concatenated sequence & static encodings into a 32-dimensional latent vector.

Saves trained weights (e.g. joint_ae.pt) in a results/ subfolder.

3.3. Step 3 – Extract latent embeddings
python src/representations/extract_embed_4.py


This script:

Loads the trained autoencoder.

Runs the encoder on train/val/test sets.

Exports a single embed.npz containing:

embed – matrix of shape (N, 32) with latent representations,

stay_id – matching identifiers.

These embeddings are the input for clustering & prediction.

3.4. Step 4 – Consensus clustering on latent embeddings
python src/clusters/cluster_consensus.py
python src/clusters/cluster_consensus_eval.py


cluster_consensus.py

Runs resampling-based consensus clustering on embed.

For K in a candidate range (default 2–9), repeatedly:

bootstraps 80% of patients,

runs k-means on the latent vectors,

accumulates a patient-by-patient consensus matrix (co-clustering frequencies).

For each K, saves:

consensus_k{k}.npy – consensus matrix,

cluster_k{k}.csv – final full-data k-means assignments,

consensus_metrics.csv – PAC, silhouette, CH, DB indices.

cluster_consensus_eval.py

Reads consensus matrices and cluster labels for all K.

Computes and plots:

CDF curves of consensus values and corresponding AUC,

PAC vs. K,

cluster-wise and item-wise consensus,

split–merge tracking of clusters across different K values.

These diagnostics are used to choose the final number of subphenotypes (e.g. K=4).

3.5. Step 5 – Train XGBoost subphenotype classifier
python src/predict/xgboost_model.py


This script:

Merges early-window summary features with the final cluster labels (chosen K).

Trains a multiclass XGBoost classifier in a one-vs-rest setting to predict subphenotype membership.

Uses cross-validation to estimate performance (e.g. AUROC, macro-F1).

Saves:

trained model artefacts,

cross-validation metrics,

(optionally) feature importance plots.

4. Target trial emulation scripts (src/ett/)

These scripts are provided for transparency in peer review.
They are intended for real ICU datasets, not the small synthetic demo.




