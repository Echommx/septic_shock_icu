#!/usr/bin/env python
# coding: utf-8
"""
build_dataset_2.py

Construct an analysis-ready dataset for the corticosteroid target trial:

- read raw static baseline table, dynamic worst-value table, treatment strategy table, and outcome table
- optionally merge subphenotype labels produced by the representation-learning and clustering pipeline
- create analysis_matrix.csv by inner-joining on stay_id and dropping non-core variables with very high missingness
- merge inverse probability weights if available
- generate descriptive tables of outcomes and covariates by treatment strategy and by subphenotype
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = Path("cortico_ett/mezxett/mg200/data_raw")
PROC_DIR = Path("cortico_ett/mezxett/mg200/data_proc/cortico")
RESULT_DIR = Path("cortico_ett/mezxett/mg200/results/ett_describe")

PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CSV = RAW_DIR / "cortico_mezxett_variable.csv"
DYN_CSV = RAW_DIR / "cortico_mezxett_dyn_all6_6h.csv"
STRATEGY_CSV = RAW_DIR / "cortico_mezxett_strategy.csv"
OUTCOME_CSV = RAW_DIR / "cortico_mezxett_outcome.csv"

LABEL_CSV = PROC_DIR / "labels_pred.csv"

ANALYSIS_CSV = PROC_DIR / "analysis_matrix.csv"
MISSING_CSV = RESULT_DIR / "missing_summary.csv"

OUTCOME_BY_STRATEGY_CSV = RESULT_DIR / "outcomes_by_strategy.csv"
COVARS_BY_SUBPHENO_CSV = RESULT_DIR / "covars_by_subphenotype.csv"
COVARS_BY_STRATEGY_CSV = RESULT_DIR / "covars_by_strategy.csv"


def _fmt_count_pct(count: float | int, denom: float | int) -> str:
    if denom is None or denom == 0 or pd.isna(denom):
        return ""
    pct = 100.0 * (float(count) / float(denom))
    return f"{int(round(count))} ({pct:.1f}%)"


def _weighted_n_map(df: pd.DataFrame, group_col: str) -> dict:
    w = pd.to_numeric(df["ipw"], errors="coerce").fillna(1.0)
    return (
        df.assign(_w=w)
        .groupby(group_col)["_w"]
        .sum()
        .round()
        .astype(int)
        .to_dict()
    )


def _require_unique_key(df: pd.DataFrame, name: str):
    if "stay_id" not in df.columns:
        raise KeyError(f"{name} 缺少 'stay_id' 列")
    if df["stay_id"].duplicated().any():
        dup = (
            df.loc[df["stay_id"].duplicated(), "stay_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{name} 的 stay_id 存在重复（示例: {dup} ...），请先去重"
        )


def _find_strategy_column(df_strategy: pd.DataFrame) -> str | None:
    candidates = [
        "strategy",
        "treat",
        "treatment",
        "group",
        "arm"
    ]
    lower_map = {c.lower(): c for c in df_strategy.columns}
    for k in candidates:
        if k in lower_map and lower_map[k].lower() != "stay_id":
            return lower_map[k]
    non_id = [c for c in df_strategy.columns if c != "stay_id"]
    if len(non_id) == 1:
        return non_id[0]
    return None


def _is_binary_numeric(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = pd.Series(series.dropna().unique())
    return set(vals.astype(int).unique()).issubset({0, 1}) and vals.nunique() <= 2


def _format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _desc_numeric(s: pd.Series) -> str:
    return f"{np.nanmean(s):.2f} ± {np.nanstd(s, ddof=1):.2f}"


def _desc_binary(s: pd.Series) -> str:
    rate = (
        100
        * (np.nansum(s) / np.sum(~pd.isna(s)))
        if np.sum(~pd.isna(s))
        else np.nan
    )
    return f"{rate:.1f}%"


def _desc_numeric_medianiqr(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna().values
    if x.size == 0:
        return ""
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    return f"{med:.2f} ({q1:.2f}–{q3:.2f})"


def _cat_chi2(df: pd.DataFrame, col: str, group_col: str) -> float:
    tab = pd.crosstab(df[group_col], df[col], dropna=True)
    if tab.size == 0 or (tab.values.sum() == 0):
        return np.nan
    try:
        _, p, _, _ = stats.chi2_contingency(tab)
        return p
    except Exception:
        return np.nan


def _num_kruskal(df: pd.DataFrame, col: str, group_col: str) -> float:
    groups = []
    for _, sub in df.groupby(group_col)[col]:
        arr = pd.to_numeric(sub, errors="coerce").dropna().values
        if len(arr) > 0:
            groups.append(arr)
    if len(groups) < 2:
        return np.nan
    try:
        _, p = stats.kruskal(*groups)
        return p
    except Exception:
        return np.nan


def _num_mannwhitney(df: pd.DataFrame, col: str, group_col: str) -> float:
    levels = [g for g in df[group_col].dropna().unique()]
    if len(levels) != 2:
        return _num_kruskal(df, col, group_col)
    a = (
        pd.to_numeric(
            df.loc[df[group_col] == levels[0], col], errors="coerce"
        )
        .dropna()
        .values
    )
    b = (
        pd.to_numeric(
            df.loc[df[group_col] == levels[1], col], errors="coerce"
        )
        .dropna()
        .values
    )
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return p
    except Exception:
        return np.nan


def main():
    static = pd.read_csv(STATIC_CSV)
    dyn = pd.read_csv(DYN_CSV)
    strategy = pd.read_csv(STRATEGY_CSV)
    outcome = pd.read_csv(OUTCOME_CSV)

    _require_unique_key(static, "静态表")
    _require_unique_key(dyn, "动态表(已最差值)")
    _require_unique_key(strategy, "策略表")
    _require_unique_key(outcome, "结局表")

    df = static.merge(dyn, on="stay_id", how="inner", suffixes=("", "_dyn"))
    df = df.merge(strategy, on="stay_id", how="inner")
    df = df.merge(outcome, on="stay_id", how="inner")

    if LABEL_CSV.exists():
        lab = pd.read_csv(LABEL_CSV)
        if "stay_id" in lab.columns:
            cols_norm = {c.strip().lower(): c for c in lab.columns}
            subp = cols_norm.get("subphenotype") or cols_norm.get("cluster")
            if subp:
                lab = lab[["stay_id", subp]].rename(
                    columns={subp: "subphenotype"}
                )
                lab = lab.drop_duplicates(subset=["stay_id"])
                df = df.merge(lab, on="stay_id", how="left")

    core_cols = {"stay_id"}
    main_strategy_col = _find_strategy_column(strategy)
    if main_strategy_col:
        core_cols.add(main_strategy_col)
    if "subphenotype" in df.columns:
        core_cols.add("subphenotype")
    outcome_cols = [c for c in outcome.columns if c != "stay_id"]
    core_cols.update(outcome_cols)

    miss_rate = df.isna().mean().rename("missing_rate").to_frame()
    miss_rate.to_csv(MISSING_CSV)
    to_drop = [
        c
        for c in df.columns
        if c not in core_cols and miss_rate.loc[c, "missing_rate"] > 0.5
    ]
    df_keep = df.drop(columns=to_drop)
    df_keep.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    WEIGHT_CSV = PROC_DIR / "balanced_cohort.csv"
    if WEIGHT_CSV.exists():
        wdf = pd.read_csv(WEIGHT_CSV)[["stay_id", "ipw"]]
        df_keep = df_keep.merge(wdf, on="stay_id", how="left")
    else:
        df_keep["ipw"] = 1.0
    df_keep["ipw"] = (
        pd.to_numeric(df_keep["ipw"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1e-8)
    )

    if main_strategy_col:
        strat_col = main_strategy_col
    else:
        strat_cols = [c for c in strategy.columns if c != "stay_id"]
        if len(strat_cols) == 0:
            strat_col = None
        else:
            strat_col = "_strategy_combo_"
            df_keep[strat_col] = df_keep[strat_cols].astype(str).agg(
                "|".join, axis=1
            )

    if strat_col:
        out_rows = []
        groups = list(df_keep[strat_col].dropna().unique())
        for oc in outcome_cols:
            s = df_keep[oc]
            is_bin = _is_binary_numeric(s)
            overall = _desc_binary(s) if is_bin else _desc_numeric(s)
            row = {"outcome": oc, "overall": overall}
            for g in groups:
                seg = df_keep.loc[df_keep[strat_col] == g, oc]
                row[str(g)] = (
                    _desc_binary(seg) if is_bin else _desc_numeric(seg)
                )
            if not is_bin:
                p = _num_kruskal(df_keep, oc, strat_col)
            else:
                p = _cat_chi2(
                    pd.DataFrame(
                        {strat_col: df_keep[strat_col], oc: s.astype("Int64")}
                    ),
                    oc,
                    strat_col,
                )
            row["p_value"] = _format_p(p)
            out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(
            OUTCOME_BY_STRATEGY_CSV, index=False, encoding="utf-8-sig"
        )

    if "subphenotype" in df_keep.columns:
        covar_cols = [
            c
            for c in df_keep.columns
            if c
            not in {"stay_id", "subphenotype"}
            | set(outcome_cols)
            | ({strat_col} if strat_col else set())
        ]
        rows = []
        for c in covar_cols:
            series = df_keep[c]
            if pd.api.types.is_numeric_dtype(series) and not _is_binary_numeric(
                series
            ):
                row = {"covariate": c}
                for g, seg in df_keep.groupby("subphenotype")[c]:
                    row[f"{g}"] = _desc_numeric_medianiqr(seg)
                row["p_value"] = _format_p(
                    _num_kruskal(df_keep, c, "subphenotype")
                )
                rows.append(row)
            else:
                series = df_keep[c]
                if pd.api.types.is_numeric_dtype(
                    series
                ) and _is_binary_numeric(series):
                    row = {"covariate": c}
                    for g, seg in df_keep.groupby("subphenotype")[c]:
                        seg_non_na = seg.dropna()
                        wseg = df_keep.loc[seg_non_na.index, "ipw"]
                        denom_w = float(wseg.sum())
                        cnt1_w = float(wseg[(seg_non_na == 1)].sum())
                        row[f"{g}"] = _fmt_count_pct(cnt1_w, denom_w)
                    row["p_value"] = _format_p(
                        _cat_chi2(
                            pd.DataFrame(
                                {
                                    "subphenotype": df_keep["subphenotype"],
                                    c: series.astype("Int64"),
                                }
                            ),
                            c,
                            "subphenotype",
                        )
                    )
                    rows.append(row)
                else:
                    seg_non_na = series.dropna()
                    levels = pd.Series(seg_non_na.unique()).tolist()
                    for lvl in levels:
                        row = {"covariate": f"{c}={lvl}"}
                        for g, sub in df_keep.groupby("subphenotype")[c]:
                            sub_non_na = sub.dropna()
                            wsub = df_keep.loc[sub_non_na.index, "ipw"]
                            denom_w = float(wsub.sum())
                            cnt_w = float(
                                wsub[(sub_non_na == lvl)].sum()
                            )
                            row[f"{g}"] = _fmt_count_pct(cnt_w, denom_w)
                        row["p_value"] = _format_p(
                            _cat_chi2(
                                pd.DataFrame(
                                    {
                                        "subphenotype": df_keep[
                                            "subphenotype"
                                        ],
                                        c: series,
                                    }
                                ),
                                c,
                                "subphenotype",
                            )
                        )
                        rows.append(row)

        tab = pd.DataFrame(rows)
        nmap = _weighted_n_map(df_keep, "subphenotype")
        if "ipw" in df_keep.columns:
            nmap = (
                df_keep.groupby("subphenotype")["ipw"]
                .sum()
                .round()
                .astype(int)
                .to_dict()
            )
        else:
            nmap = (
                df_keep["subphenotype"]
                .value_counts(dropna=False)
                .to_dict()
            )

        tab.rename(
            columns={
                str(k): f"{k} (n={nmap.get(k, 0)})" for k in nmap
            },
            inplace=True,
        )
        tab.to_csv(
            COVARS_BY_SUBPHENO_CSV, index=False, encoding="utf-8-sig"
        )

    if strat_col:
        covar_cols = [
            c
            for c in df_keep.columns
            if c
            not in {"stay_id", "subphenotype", strat_col}
            | set(outcome_cols)
        ]
        rows = []
        for c in covar_cols:
            series = df_keep[c]
            if pd.api.types.is_numeric_dtype(series) and not _is_binary_numeric(
                series
            ):
                row = {"covariate": c}
                for g, seg in df_keep.groupby(strat_col)[c]:
                    row[f"{g}"] = _desc_numeric_medianiqr(seg)
                row["p_value"] = _format_p(
                    _num_kruskal(df_keep, c, strat_col)
                )
                rows.append(row)
            else:
                if pd.api.types.is_numeric_dtype(
                    series
                ) and _is_binary_numeric(series):
                    row = {"covariate": c}
                    for g, seg in df_keep.groupby(strat_col)[c]:
                        seg_non_na = seg.dropna()
                        wseg = df_keep.loc[seg_non_na.index, "ipw"]
                        denom_w = float(wseg.sum())
                        cnt1_w = float(
                            wseg[(seg_non_na == 1)].sum()
                        )
                        row[f"{g}"] = _fmt_count_pct(cnt1_w, denom_w)
                    row["p_value"] = _format_p(
                        _cat_chi2(
                            df_keep[[strat_col, c]].assign(
                                **{c: series.astype("Int64")}
                            ),
                            c,
                            strat_col,
                        )
                    )
                    rows.append(row)
                else:
                    levels = (
                        pd.Series(series.dropna().unique())
                        .tolist()
                    )
                    for lvl in levels:
                        row = {"covariate": f"{c}={lvl}"}
                        for g, sub in df_keep.groupby(strat_col)[c]:
                            sub_non_na = sub.dropna()
                            wsub = df_keep.loc[sub_non_na.index, "ipw"]
                            denom_w = float(wsub.sum())
                            cnt_w = float(
                                wsub[(sub_non_na == lvl)].sum()
                            )
                            row[f"{g}"] = _fmt_count_pct(cnt_w, denom_w)
                        row["p_value"] = _format_p(
                            _cat_chi2(
                                pd.DataFrame(
                                    {
                                        strat_col: df_keep[strat_col],
                                        c: series,
                                    }
                                ),
                                c,
                                strat_col,
                            )
                        )
                        rows.append(row)

        tab = pd.DataFrame(rows)
        nmap = _weighted_n_map(df_keep, strat_col)
        if "ipw" in df_keep.columns:
            nmap = (
                df_keep.groupby(strat_col)["ipw"]
                .sum()
                .round()
                .astype(int)
                .to_dict()
            )
        else:
            nmap = (
                df_keep[strat_col]
                .value_counts(dropna=False)
                .to_dict()
            )

        tab.rename(
            columns={
                str(k): f"{k} (n={nmap.get(k, 0)})" for k in nmap
            },
            inplace=True,
        )
        tab.to_csv(
            COVARS_BY_STRATEGY_CSV, index=False, encoding="utf-8-sig"
        )

    print("✔ 分析矩阵：", ANALYSIS_CSV)
    print("✔ 缺失概览：", MISSING_CSV)
    if strat_col:
        print("✔ 结局×策略：", OUTCOME_BY_STRATEGY_CSV)
        print("✔ 协变量×策略：", COVARS_BY_STRATEGY_CSV)
    if "subphenotype" in df_keep.columns:
        print("✔ 协变量×亚型：", COVARS_BY_SUBPHENO_CSV)
    print("🎉 Done – see", PROC_DIR, "and", RESULT_DIR)


if __name__ == "__main__":
    main()
