#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_balanced_cohort.py v5

Estimate propensity scores using multivariable logistic regression, compute
stabilized inverse probability of treatment weights (IPTW) with truncation,
assess covariate balance using standardized mean differences (SMD), and
export a weighted cohort, representative resampled cohort, balance tables,
Love plots, and weighted outcome summaries for the corticosteroid target trial.
"""

from __future__ import annotations
import warnings
import sys
import pathlib
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DP = pathlib.Path("cortico_ett/mezxett/mg200/data_proc/cortico")
RS = pathlib.Path("cortico_ett/mezxett/mg200/results/ett")
DP.mkdir(parents=True, exist_ok=True)
RS.mkdir(parents=True, exist_ok=True)

IN_FILE = DP / "analysis_matrix.csv"
COHORT_CSV = DP / "balanced_cohort.csv"
SMD_CSV = RS / "balance_smd_full.csv"
LOVE_CONT = RS / "love_plot_continuous.png"
LOVE_CAT = RS / "love_plot_categorical.png"
OUTCOME_CSV = RS / "weighted_outcomes.csv"
OUTCOME_STRAT_CSV = RS / "weighted_outcomes_strategy.csv"
WEIGHTED_REP_CSV = DP / "weighted_cohort_representative.csv"

CAT_VARS: List[str] = [
    "age_bin",
    "sex",
    "ed_admit",
    "hx_htn",
    "hx_dm",
    "hx_chf",
    "hx_ckd",
    "hx_cirrhosis",
    "site_pulm",
    "site_abd",
    "site_uri",
    "site_blood",
]

CONT_VARS = [
    "sbp_min",
    "dbp_min",
    "hr_max",
    "rr_max",
    "sao2_min",
    "map_min",
    "lactate_max",
    "creat_max",
    "na_max",
    "albumin_max",
    "band_max",
    "k_max"
    "plt_min",
    "wbc_max",
    "bun_max",
    "cl_max",
    "bicarb_max",
    "glucose_max",
    "be_min",
    "inr_max",
    "pao2_min",
    "paco2_max",
    "ph_min",
    "temp_max",
    "bili_max",
    "neu_max",
    "lym_max",
    "ast_max",
    "alt_max",
]

OUTCOME_VARS = ["d28_mort", "icu_mort", "icu_los", "vent_days", "crrt_days"]

CUT_SEQ = [(0.01, 0.99), (0.05, 0.95), (0.10, 0.90)]
SMD_THR = 0.10


def smd_cont(x_t: np.ndarray, x_c: np.ndarray) -> float:
    sd_p = np.sqrt((x_t.var(ddof=0) + x_c.var(ddof=0)) / 2)
    return 0.0 if sd_p == 0 else (x_t.mean() - x_c.mean()) / sd_p


def smd_cat(p_t: float, p_c: float) -> float:
    sd_p = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    return 0.0 if sd_p == 0 else (p_t - p_c) / sd_p


df = pd.read_csv(IN_FILE).rename(columns={"strategy": "treat"})
df["treat"] = df["treat"].astype(int)

n_total = len(df)
n_treat0 = (df.treat == 0).sum()
n_treat1 = (df.treat == 1).sum()

for c in CONT_VARS:
    df[c].fillna(df[c].median(), inplace=True)


def fillna_for_categoricals(df_in: pd.DataFrame, cat_vars: list[str]) -> pd.DataFrame:
    """Fill missing categorical values with 0 and keep them as 0/1-like indicators."""
    na = df_in[cat_vars].isna().sum()
    if (na > 0).any():
        print("[INFO] Missing values in categorical variables were filled with 0:")
        print(na[na > 0])
        df_in.loc[:, cat_vars] = df_in[cat_vars].fillna(0)
    return df_in


df = fillna_for_categoricals(df, CAT_VARS)

na_cat = df[CAT_VARS].isna().sum()
if na_cat.any():
    miss = na_cat[na_cat > 0]
    sys.exit(f"[ERROR] Missing values remained in categorical variables:\n{miss}")


def calc_weight(
    data: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> tuple[np.ndarray, ColumnTransformer]:
    """Estimate stabilized IPTW with group-wise truncation from a logistic PS model."""
    pipe = ColumnTransformer(
        [
            ("cont", Pipeline([("scaler", StandardScaler())]), CONT_VARS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_VARS,
            ),
        ]
    )
    X = pipe.fit_transform(data)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X, data["treat"])
    ps = lr.predict_proba(X)[:, 1].clip(1e-3, 1 - 1e-3)
    p_t = data["treat"].mean()
    w = np.where(data.treat == 1, p_t / ps, (1 - p_t) / (1 - ps))
    w_new = w.copy()
    for g in [0, 1]:
        mask = (data.treat == g).values
        if mask.any():
            w_group = w[mask]
            lo, hi = np.quantile(w_group, [q_low, q_high])
            w_new[mask] = np.clip(w_group, lo, hi)
    return w_new, pipe


def format_count_percent(count: float | int, total: float | int, percent: float) -> str:
    """Format unweighted count and percentage as 'n (p%)'."""
    return f"{int(count)} ({percent:.1f}%)"


def format_weighted_count_percent(
    weighted_count: float, total_weight: float, percent: float
) -> str:
    """Format weighted count and percentage as 'n_w (p%)'."""
    return f"{weighted_count:.0f} ({percent:.1f}%)"


def _w_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """
    Weighted quantile (including weighted median).

    x: numeric values
    w: corresponding positive weights
    q: quantile in [0, 1], e.g., 0.5=median, 0.25, 0.75 for IQR
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan

    x = x[m]
    w = w[m]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return np.nan
    return float(np.interp(q * cw[-1], cw, x))


def smd_full(data: pd.DataFrame, w: np.ndarray, pipe: ColumnTransformer) -> pd.DataFrame:
    """Compute raw and weighted SMD and summary statistics for all covariates."""
    rec: list[list[object]] = []

    total_weight = w.sum()
    weight_t0 = w[data.treat == 0].sum()
    weight_t1 = w[data.treat == 1].sum()

    for c in CONT_VARS:
        t = data.loc[data.treat == 1, c]
        c0 = data.loc[data.treat == 0, c]
        smd_b = smd_cont(t.values, c0.values)

        wt_t = float(np.average(t, weights=w[data.treat == 1]))
        wt_c = float(np.average(c0, weights=w[data.treat == 0]))

        total_mean = float(np.average(data[c], weights=w))
        total_var = float(np.average((data[c] - total_mean) ** 2, weights=w))
        _ = np.sqrt(total_var)

        vt = float(np.average((t - wt_t) ** 2, weights=w[data.treat == 1]))
        vc = float(np.average((c0 - wt_c) ** 2, weights=w[data.treat == 0]))
        sd_p = np.sqrt((vt + vc) / 2)
        smd_a = 0.0 if sd_p == 0 else (wt_t - wt_c) / sd_p

        med_all = float(np.nanmedian(data[c]))
        q1_all = float(np.nanpercentile(data[c], 25))
        q3_all = float(np.nanpercentile(data[c], 75))

        med_c0 = float(np.nanmedian(c0))
        q1_c0 = float(np.nanpercentile(c0, 25))
        q3_c0 = float(np.nanpercentile(c0, 75))

        med_t1 = float(np.nanmedian(t))
        q1_t1 = float(np.nanpercentile(t, 25))
        q3_t1 = float(np.nanpercentile(t, 75))

        wt_total_med = _w_quantile(data[c].values, w, 0.5)
        wt_total_q1 = _w_quantile(data[c].values, w, 0.25)
        wt_total_q3 = _w_quantile(data[c].values, w, 0.75)

        wt_c0_med = _w_quantile(c0.values, w[data.treat == 0], 0.5)
        wt_c0_q1 = _w_quantile(c0.values, w[data.treat == 0], 0.25)
        wt_c0_q3 = _w_quantile(c0.values, w[data.treat == 0], 0.75)

        wt_t1_med = _w_quantile(t.values, w[data.treat == 1], 0.5)
        wt_t1_q1 = _w_quantile(t.values, w[data.treat == 1], 0.25)
        wt_t1_q3 = _w_quantile(t.values, w[data.treat == 1], 0.75)

        rec.append(
            [
                c,
                "continuous",
                f"{med_all:.2f} ({q1_all:.2f}–{q3_all:.2f})",
                f"{med_c0:.2f} ({q1_c0:.2f}–{q3_c0:.2f})",
                f"{med_t1:.2f} ({q1_t1:.2f}–{q3_t1:.2f})",
                smd_b,
                f"{wt_total_med:.2f} ({wt_total_q1:.2f}–{wt_total_q3:.2f})",
                f"{wt_c0_med:.2f} ({wt_c0_q1:.2f}–{wt_c0_q3:.2f})",
                f"{wt_t1_med:.2f} ({wt_t1_q1:.2f}–{wt_t1_q3:.2f})",
                smd_a,
            ]
        )

    ohe: OneHotEncoder = pipe.named_transformers_["cat"]  # type: ignore[assignment]
    cat_X = ohe.transform(df[CAT_VARS])
    cat_cols = ohe.get_feature_names_out(CAT_VARS)
    dummies = pd.DataFrame(cat_X, columns=cat_cols, index=data.index)

    for col in dummies.columns:
        parts = col.rsplit("_", 1)
        name = parts[0]
        level = parts[1] if len(parts) > 1 else "1"

        series = dummies[col]
        s_treat0 = series[data.treat == 0]
        s_treat1 = series[data.treat == 1]

        count_all = float(series.sum())
        count_t0 = float(s_treat0.sum())
        count_t1 = float(s_treat1.sum())
        p_all = float(series.mean() * 100)
        p_t0 = float(s_treat0.mean() * 100)
        p_t1 = float(s_treat1.mean() * 100)
        smd_b = smd_cat(p_t1 / 100.0, p_t0 / 100.0)

        w_all = float(np.sum(series * w))
        w_t0 = float(np.sum(s_treat0 * w[data.treat == 0]))
        w_t1 = float(np.sum(s_treat1 * w[data.treat == 1]))

        p_all_w = float(np.average(series, weights=w) * 100)
        p_t0_w = float(np.average(s_treat0, weights=w[data.treat == 0]) * 100)
        p_t1_w = float(np.average(s_treat1, weights=w[data.treat == 1]) * 100)
        smd_a = smd_cat(p_t1_w / 100.0, p_t0_w / 100.0)

        rec.append(
            [
                f"{name}={level}",
                "categorical",
                format_count_percent(count_all, n_total, p_all),
                format_count_percent(count_t0, n_treat0, p_t0),
                format_count_percent(count_t1, n_treat1, p_t1),
                smd_b,
                format_weighted_count_percent(w_all, total_weight, p_all_w),
                format_weighted_count_percent(w_t0, weight_t0, p_t0_w),
                format_weighted_count_percent(w_t1, weight_t1, p_t1_w),
                smd_a,
            ]
        )

    columns = [
        "variable",
        "type",
        f"raw_total (n={n_total})",
        f"raw_str0 (n={n_treat0})",
        f"raw_str1 (n={n_treat1})",
        "raw_smd",
        f"wt_total (n={total_weight:.0f})",
        f"wt_str0 (n={weight_t0:.0f})",
        f"wt_str1 (n={weight_t1:.0f})",
        "wt_smd",
    ]

    return pd.DataFrame(rec, columns=columns)


print("▶ IPTW 迭代")
final: tuple[np.ndarray, ColumnTransformer, pd.DataFrame] | None = None
for i, (q_low, q_high) in enumerate(CUT_SEQ, 1):
    w, pipe = calc_weight(df, q_low, q_high)
    smd_tab = smd_full(df, w, pipe)
    worst = float(smd_tab["wt_smd"].abs().max())
    print(f"  • {int(q_low*100)}–{int(q_high*100)} %  max|SMD|={worst:.3f}")
    final = (w, pipe, smd_tab)
    if worst < SMD_THR or i == len(CUT_SEQ):
        break

if final is None:
    raise RuntimeError("IPTW computation failed to produce any result.")

w, pipe, smd_tab = final

ratio = float(np.nanmax(w) / max(float(np.nanmin(w[w > 0])), 1e-12))
if ratio > 50:
    warnings.warn(
        f"High weight variability (max/min={ratio:.1f}). Consider stronger truncation."
    )

df_out = df.copy()
df_out["ipw"] = w
df_out.to_csv(COHORT_CSV, index=False, encoding="utf-8-sig")
print("✓ balanced_cohort.csv")

weighted_cohort = df.copy()
weighted_cohort["ipw"] = w

total_weight = float(w.sum())
rng = np.random.default_rng(1)
resampled_indices = rng.choice(
    df.index,
    size=int(round(total_weight)),
    replace=True,
    p=w / w.sum(),
)

rep_cohort = df.loc[resampled_indices].copy()
rep_cohort["sample_type"] = "resampled"
rep_cohort["ipw"] = 1.0
rep_cohort.to_csv(WEIGHTED_REP_CSV, index=False, encoding="utf-8-sig")
print(f"✓ 代表性加权队列已保存 (n={len(rep_cohort)} ≈ {total_weight:.0f}): {WEIGHTED_REP_CSV}")

smd_tab.to_csv(SMD_CSV, index=False, encoding="utf-8-sig")
print(f"✓ {SMD_CSV} (包含样本量和具体数值)")

print("▶ 创建分开的Love-plot")

cont_tab = smd_tab[smd_tab["type"] == "continuous"]
if not cont_tab.empty:
    plt.figure(figsize=(8, max(6, len(cont_tab) * 0.3)))
    order = cont_tab.sort_values("raw_smd", key=abs, ascending=False)["variable"]
    plt.scatter(cont_tab["raw_smd"], order, c="red", label="Raw")
    plt.scatter(cont_tab["wt_smd"], order, c="blue", label="Weighted")
    plt.axvline(0, color="k")
    plt.axvline(SMD_THR, color="grey", ls="--")
    plt.axvline(-SMD_THR, color="grey", ls="--")
    plt.xlabel("Standardised Mean Difference")
    plt.title("Continuous Variables")
    plt.tight_layout()
    plt.legend()
    plt.savefig(LOVE_CONT, dpi=300)
    print(f"✓ {LOVE_CONT}")
else:
    print("⚠️ 无连续变量数据，跳过连续变量Love-plot")

cat_tab = smd_tab[smd_tab["type"] == "categorical"]
if not cat_tab.empty:
    plt.figure(figsize=(8, max(6, len(cat_tab) * 0.3)))
    order = cat_tab.sort_values("raw_smd", key=abs, ascending=False)["variable"]
    plt.scatter(cat_tab["raw_smd"], order, c="red", label="Raw")
    plt.scatter(cat_tab["wt_smd"], order, c="blue", label="Weighted")
    plt.axvline(0, color="k")
    plt.axvline(SMD_THR, color="grey", ls="--")
    plt.axvline(-SMD_THR, color="grey", ls="--")
    plt.xlabel("Standardised Mean Difference")
    plt.title("Categorical Variables")
    plt.tight_layout()
    plt.legend()
    plt.savefig(LOVE_CAT, dpi=300)
    print(f"✓ {LOVE_CAT}")
else:
    print("⚠️ 无分类变量数据，跳过分类变量Love-plot")


def wt_prop(s: pd.Series, wt: np.ndarray) -> float:
    return float(np.average(s, weights=wt))


def wt_mean(s: pd.Series, wt: np.ndarray) -> float:
    return float(np.average(s, weights=wt))


def wt_sd(s: pd.Series, wt: np.ndarray, mu: float) -> float:
    return float(np.sqrt(np.average((s - mu) ** 2, weights=wt)))


print("▶ 生成加权结局表...")

rows: list[list[object]] = []
rows_strat: list[list[object]] = []

for grp in ["all", 0, 1, 2, 3]:
    if grp == "all":
        mask = np.ones(len(df_out), dtype=bool)
        group_name = "All"
    else:
        mask = df_out["subphenotype"] == grp
        group_name = f"S{grp}"

    wt = w[mask]
    weighted_n = float(wt.sum())
    n_eff = float(weighted_n**2 / (wt**2).sum())

    stat = [group_name, int(weighted_n), int(n_eff)]

    for v in ["d28_mort", "icu_mort"]:
        stat.append(f"{wt_prop(df_out.loc[mask, v], wt) * 100:.1f}%")

    for v in ["icu_los", "vent_days", "crrt_days"]:
        mu = wt_mean(df_out.loc[mask, v], wt)
        sd = wt_sd(df_out.loc[mask, v], wt, mu)
        stat.append(f"{mu:.1f}±{sd:.1f}")

    rows.append(stat)

    for strategy in [0, 1]:
        strat_mask = mask & (df_out.treat == strategy)
        wt_strat = w[strat_mask]

        if wt_strat.sum() > 0:
            weighted_n_strat = float(wt_strat.sum())
            n_eff_strat = float(weighted_n_strat**2 / (wt_strat**2).sum())

            stat_strat = [
                group_name,
                f"Strategy {strategy}",
                int(weighted_n_strat),
                int(n_eff_strat),
            ]

            for v in ["d28_mort", "icu_mort"]:
                stat_strat.append(
                    f"{wt_prop(df_out.loc[strat_mask, v], wt_strat) * 100:.1f}%"
                )

            for v in ["icu_los", "vent_days", "crrt_days"]:
                mu = wt_mean(df_out.loc[strat_mask, v], wt_strat)
                sd = wt_sd(df_out.loc[strat_mask, v], wt_strat, mu)
                stat_strat.append(f"{mu:.1f}±{sd:.1f}")

            rows_strat.append(stat_strat)

pd.DataFrame(
    rows,
    columns=[
        "group",
        "Weighted_N",
        "N_eff",
        "d28_mort",
        "icu_mort",
        "icu_los",
        "vent_days",
        "crrt_days",
    ],
).to_csv(OUTCOME_CSV, index=False, encoding="utf-8-sig")
print(f"✓ {OUTCOME_CSV} (按组统计，包含两种样本量)")

pd.DataFrame(
    rows_strat,
    columns=[
        "group",
        "strategy",
        "Weighted_N",
        "N_eff",
        "d28_mort",
        "icu_mort",
        "icu_los",
        "vent_days",
        "crrt_days",
    ],
).to_csv(OUTCOME_STRAT_CSV, index=False, encoding="utf-8-sig")
print(f"✓ {OUTCOME_STRAT_CSV} (按组+策略统计，包含两种样本量)")

print("\n📝 加权样本量说明:")
print(
    f" • 原始样本量: All = {n_total}, Strategy 0 = {n_treat0}, Strategy 1 = {n_treat1}"
)
print(
    f" • 加权后样本量 (Weighted_N): All = {w.sum():.0f}, "
    f"Strategy 0 = {w[df.treat == 0].sum():.0f}, "
    f"Strategy 1 = {w[df.treat == 1].sum():.0f}"
)
print(
    " • 有效样本量 (N_eff): "
    f"All = {w.sum() ** 2 / (w ** 2).sum():.0f}, "
    f"Strategy 0 = {w[df.treat == 0].sum() ** 2 / (w[df.treat == 0] ** 2).sum():.0f}, "
    f"Strategy 1 = {w[df.treat == 1].sum() ** 2 / (w[df.treat == 1] ** 2).sum():.0f}"
)
print(" • 加权后样本量变化是 IPTW 方法的正常结果，因为权重调整改变了样本的统计代表性")

print("\n✅ 全部完成!  文件位置:")
print(" •", COHORT_CSV)
print(" •", SMD_CSV)
print(" •", LOVE_CONT)
print(" •", LOVE_CAT)
print(" •", OUTCOME_CSV)
print(" •", OUTCOME_STRAT_CSV)
