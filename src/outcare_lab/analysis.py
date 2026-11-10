"""Statistical analysis for the OutCaRe Outcomes Lab.

All outputs are aggregates (counts, means, model coefficients). No row level data
is written to tables, figures or reports.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.imputation.mice import MICEData
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

SMALL_CELL = 5  # counts below this are shown as "<5" in categorical tables

CONTINUOUS_PAIRS = [
    ("6MWD (m)", "t06mwd", "t16mwd", "higher"),
    ("Endurance training load (W)", "t0training", "t1training", "higher"),
    ("Maximum exercise capacity (W)", "t0maxcap", "t1maxcap", "higher"),
    ("Systolic BP (mmHg)", "t0sys", "t1sys", "lower"),
    ("Diastolic BP (mmHg)", "t0dia", "t1dia", "lower"),
    ("LDL cholesterol (mmol/l)", "t0ldl", "t1ldl", "lower"),
    ("PHQ-9 depression score", "t0phqsum", "t1phqsum", "lower"),
    ("HAF-17 heart-focused anxiety (mean)", "t0hafmean", "t1hafmean", "lower"),
    ("WHO-5 well-being", "t0who5", "t1who5", "higher"),
    ("SF-12 PCS", "t0pcs12", "t1pcs12", "higher"),
    ("SF-12 MCS", "t0mcs12", "t1mcs12", "higher"),
    ("IRES-24 physical health", "t0ires24phys", "t1ires24phys", "higher"),
    ("IRES-24 subjective health", "t0ires24subj", "t1ires24subj", "higher"),
    ("IRES-24 pain (higher = less pain)", "t0ires24pain", "t1ires24pain", "higher"),
    ("ASKU self-efficacy", "t0askumean", "t1askumean", "higher"),
]

BINARY_PAIRS = [
    ("Current smoker", "t0smoker_yes", "t1smoker_yes"),
    ("Lifestyle change motivation", "t0lifestyle_yes", "t1lifestyle_yes"),
    ("Excellent/very good health prognosis", "t0health_yes", "t1health_yes"),
    ("Pension desire", "t0pension_yes", "t1pension_yes"),
    ("Negative occupational prognosis", "t0occupat_yes", "t1occupat_yes"),
]

INDICATION_LABELS = {
    1: "ACS", 2: "Arterial hypertension", 3: "Chronic heart failure",
    4: "Heart valve disease", 5: "Stable CAD", 6: "Myo-/endo-/pericarditis",
    7: "Cardiac arrhythmia", 8: "Atherosclerosis incl. PAD", 9: "Aortic diseases",
    10: "Venous disease", 11: "Intervention (PCI, ICD/CRT)", 12: "CABG", 13: "Other",
}

# Pre-specified predictor set, fixed before looking at any outcome association.
# Chosen to mirror the domains in the paper's final models that exist in the open file.
# Scaling: age per 10 years; training per 10 W; SF-12 and WHO-5 per 10 points.
PREDICTORS = [
    ("age10", "Age (per 10 years)"),
    ("female", "Female sex"),
    ("edu_high", "College/university education"),
    ("acs", "Indication: ACS"),
    ("hf_indication", "Indication: chronic heart failure"),
    ("training0_10", "Training load T0 (per 10 W)"),
    ("dtraining_10", "Change in training load T0 to T1 (per 10 W)"),
    ("pcs0_10", "SF-12 PCS T0 (per 10 pts)"),
    ("dpcs_10", "Change in PCS T0 to T1 (per 10 pts)"),
    ("mcs0_10", "SF-12 MCS T0 (per 10 pts)"),
    ("dmcs_10", "Change in MCS T0 to T1 (per 10 pts)"),
    ("haf0", "HAF-17 mean T0 (per 1 pt)"),
    ("dhaf", "Change in HAF-17 T0 to T1 (per 1 pt)"),
    ("who0_10", "WHO-5 T0 (per 10 pts)"),
    ("asku0", "ASKU self-efficacy T0 (per 1 pt)"),
    ("t1health_yes", "Excellent/very good health prognosis T1"),
    ("t1pension_yes", "Pension desire T1"),
    ("t1occupat_yes", "Negative occupational prognosis T1"),
]
PRED_NAMES = [p for p, _ in PREDICTORS]
PRED_LABELS = dict(PREDICTORS)

# Raw columns used by the imputation model (predictors are derived from these after imputation).
IMPUTATION_COLUMNS = [
    "age", "female", "edu_high", "acs", "hf_indication", "outpatient", "chronic_admission",
    "t0training", "t1training", "t0pcs12", "t1pcs12", "t0mcs12", "t1mcs12",
    "t0hafmean", "t1hafmean", "t0who5", "t1who5", "t0askumean", "t0phqsum",
    "t0health_yes", "t1health_yes", "t0pension_yes", "t1pension_yes",
    "t0occupat_yes", "t1occupat_yes", "sickleave_yes", "occupation_yes",
    "rtw_yes", "t2pcs12", "t2mcs12",
]


def fmt_n(n: int) -> str:
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else str(int(n))


def wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def mean_ci(x: pd.Series) -> tuple[float, float, float, int]:
    x = x.dropna()
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - t * se), float(m + t * se), n


# ---------------------------------------------------------------- descriptives

def baseline_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(d)

    def cont(label, col):
        x = d[col].dropna()
        rows.append({"variable": label, "n_available": len(x),
                     "summary": f"{x.mean():.1f} (SD {x.std():.1f}); median {x.median():.0f}; range {x.min():.0f} to {x.max():.0f}"})

    def binary(label, col):
        x = d[col].dropna()
        k = int(x.sum())
        rows.append({"variable": label, "n_available": len(x),
                     "summary": f"{fmt_n(k)} ({100 * k / len(x):.1f}%)"})

    cont("Age (years)", "age")
    k = int((d["sex"] == 1).sum())
    rows.append({"variable": "Male sex", "n_available": n, "summary": f"{k} ({100 * k / n:.1f}%)"})
    binary("Female sex", "female")
    edu = d["education"].dropna()
    for code, lab in {1: "<10th grade", 2: "Secondary school", 3: "College/University", 4: "Other"}.items():
        k = int((edu == code).sum())
        rows.append({"variable": f"Education: {lab}", "n_available": len(edu),
                     "summary": f"{fmt_n(k)} ({100 * k / len(edu):.1f}%)"})
    liv = d["living"].dropna()
    k = int((liv == 1).sum())
    rows.append({"variable": "Living with family/partner", "n_available": len(liv),
                 "summary": f"{k} ({100 * k / len(liv):.1f}%)"})
    binary("Employed before CR", "occupation_yes")
    binary("Sick leave before CR", "sickleave_yes")
    k = int((d["cradmission"] == 1).sum())
    rows.append({"variable": "Admission after acute event", "n_available": n,
                 "summary": f"{k} ({100 * k / n:.1f}%)"})
    k = int((d["crsetting"] == 1).sum())
    rows.append({"variable": "Inpatient CR", "n_available": n, "summary": f"{k} ({100 * k / n:.1f}%)"})
    cont("CR stay (days)", "crstay")
    for code, lab in {1: "normal", 2: "premature, own request", 3: "transferred to acute care"}.items():
        k = int((d["entlart"] == code).sum())
        rows.append({"variable": f"Discharge: {lab}", "n_available": n,
                     "summary": f"{fmt_n(k)} ({100 * k / n:.1f}%)"})
    for code, lab in INDICATION_LABELS.items():
        k = int((d["indication"] == code).sum())
        rows.append({"variable": f"Indication: {lab}", "n_available": n,
                     "summary": f"{fmt_n(k)} ({100 * k / n:.1f}%)"})
    cont("Number of comorbidities/risk factors", "comorb")
    for col, lab in [("arthyp_yes", "Hypertension"), ("hyperlip_yes", "Hyperlipidaemia"),
                     ("diab_yes", "Diabetes"), ("atrialfib_yes", "Atrial fibrillation"),
                     ("pad_yes", "PAD"), ("depression_yes", "Depression (comorbidity)"),
                     ("copd_yes", "COPD"), ("kidney_yes", "Kidney disease")]:
        binary(f"Comorbidity: {lab}", col)
    binary("Current smoker T0", "t0smoker_yes")
    return pd.DataFrame(rows)


def missingness_table(d: pd.DataFrame, raw_cols) -> pd.DataFrame:
    out = pd.DataFrame({"variable": list(raw_cols)})
    out["n_available"] = [int(d[c].notna().sum()) for c in raw_cols]
    out["pct_missing"] = [round(100 * d[c].isna().mean(), 1) for c in raw_cols]
    return out


def continuous_change(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, a, b, better in CONTINUOUS_PAIRS:
        both = d[[a, b]].dropna()
        diff = both[b] - both[a]
        m, lo, hi, n = mean_ci(diff)
        sd0 = both[a].std(ddof=1)
        try:
            p = stats.wilcoxon(both[b], both[a], zero_method="wilcox").pvalue
        except ValueError:
            p = np.nan
        rows.append({
            "measure": label, "n_t0": int(d[a].notna().sum()), "n_t1": int(d[b].notna().sum()),
            "n_paired": n, "mean_t0": both[a].mean(), "sd_t0": sd0, "mean_t1": both[b].mean(),
            "sd_t1": both[b].std(ddof=1), "mean_change": m, "ci_low": lo, "ci_high": hi,
            "median_change": diff.median(), "ses": m / sd0, "wilcoxon_p": p,
            "better_direction": better,
        })
    return pd.DataFrame(rows)


def binary_change(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, a, b in BINARY_PAIRS:
        both = d[[a, b]].dropna()
        n = len(both)
        p0, p1 = both[a].mean(), both[b].mean()
        tab = pd.crosstab(both[a], both[b]).reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0)
        p = mcnemar(tab.values, exact=False, correction=True).pvalue
        rows.append({"measure": label, "n_paired": n, "pct_t0": 100 * p0, "pct_t1": 100 * p1,
                     "diff_pp": 100 * (p1 - p0), "mcnemar_p": p})
    return pd.DataFrame(rows)


def followup_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, lab in [("rtw_yes", "Returned to work"), ("t2retired_yes", "Retired"),
                     ("t2pension_yes", "Applied for/desire pension"),
                     ("t2unemployed_yes", "Unemployed"), ("t2sick_yes", "On sick leave"),
                     ("t2hospital_yes", "Readmission to acute hospital care"),
                     ("t2health_yes", "Health moderately/greatly improved"),
                     ("t2capacity_yes", "Occupational/physical capacity enhanced")]:
        x = d[col].dropna()
        k, n = int(x.sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append({"outcome_6_months": lab, "n_available": n, "count": k,
                     "pct": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi})
    for col, lab in [("t2pcs12", "SF-12 PCS at 6 months"), ("t2mcs12", "SF-12 MCS at 6 months")]:
        m, lo, hi, n = mean_ci(d[col])
        rows.append({"outcome_6_months": lab, "n_available": n, "count": np.nan,
                     "pct": np.nan, "mean": m, "sd": d[col].std(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def rtw_by_factor(d: pd.DataFrame) -> pd.DataFrame:
    factors = [
        ("t1pension_yes", "Pension desire T1", {1.0: "yes", 0.0: "no"}),
        ("t1occupat_yes", "Negative occupational prognosis T1", {1.0: "yes", 0.0: "no"}),
        ("edu_high", "Education", {1.0: "college/university", 0.0: "other"}),
        ("acs", "Indication", {1.0: "ACS", 0.0: "other"}),
        ("hf_indication", "Heart failure indication", {1.0: "yes", 0.0: "no"}),
        ("female", "Sex", {1.0: "female", 0.0: "male"}),
        ("sickleave_yes", "Sick leave before CR", {1.0: "yes", 0.0: "no"}),
        ("t1health_yes", "Excellent/very good health prognosis T1", {1.0: "yes", 0.0: "no"}),
    ]
    rows = []
    for col, lab, levels in factors:
        for val, lev in levels.items():
            sub = d.loc[d[col] == val, "rtw_yes"]
            k, n = int(sub.sum()), len(sub)
            lo, hi = wilson(k, n)
            rows.append({"factor": lab, "level": lev, "n": n, "rtw_n": fmt_n(k),
                         "rtw_pct": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi})
    return pd.DataFrame(rows)


def selection_6mwd(d: pd.DataFrame) -> pd.DataFrame:
    has = d[["t06mwd", "t16mwd"]].notna().all(axis=1)
    rows = []
    for lab, col, kind in [("Age (years)", "age", "mean"), ("Female (%)", "female", "pct"),
                           ("Inpatient CR (%)", "outpatient", "inv_pct"),
                           ("ACS indication (%)", "acs", "pct"),
                           ("Returned to work (%)", "rtw_yes", "pct"),
                           ("SF-12 PCS T0", "t0pcs12", "mean"),
                           ("Training load T0 (W)", "t0training", "mean")]:
        vals = []
        for g in (has, ~has):
            x = d.loc[g, col].dropna()
            if kind == "mean":
                vals.append(f"{x.mean():.1f}")
            elif kind == "pct":
                vals.append(f"{100 * x.mean():.1f}")
            else:
                vals.append(f"{100 * (1 - x.mean()):.1f}")
        rows.append({"characteristic": lab, "both_6mwd": vals[0], "not_both_6mwd": vals[1]})
    rows.insert(0, {"characteristic": "n", "both_6mwd": str(int(has.sum())),
                    "not_both_6mwd": str(int((~has).sum()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models

def add_model_terms(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age10"] = d["age"] / 10
    d["training0_10"] = d["t0training"] / 10
    d["dtraining_10"] = (d["t1training"] - d["t0training"]) / 10
    d["pcs0_10"] = d["t0pcs12"] / 10
    d["dpcs_10"] = (d["t1pcs12"] - d["t0pcs12"]) / 10
    d["mcs0_10"] = d["t0mcs12"] / 10
    d["dmcs_10"] = (d["t1mcs12"] - d["t0mcs12"]) / 10
    d["haf0"] = d["t0hafmean"]
    d["dhaf"] = d["t1hafmean"] - d["t0hafmean"]
    d["who0_10"] = d["t0who5"] / 10
    d["asku0"] = d["t0askumean"]
    return d


def _fit(df: pd.DataFrame, outcome: str, kind: str):
    X = sm.add_constant(df[PRED_NAMES].astype(float))
    y = df[outcome].astype(float)
    fam = sm.families.Binomial() if kind == "logit" else sm.families.Gaussian()
    return sm.GLM(y, X, family=fam).fit()


def _tidy(params, se, df_t, kind, method, n, outcome):
    if df_t is None:
        crit = stats.norm.ppf(0.975)
        p = 2 * stats.norm.sf(np.abs(params / se))
    else:
        crit = stats.t.ppf(0.975, df_t)
        p = 2 * stats.t.sf(np.abs(params / se), df_t)
    lo, hi = params - crit * se, params + crit * se
    out = pd.DataFrame({"term": params.index, "estimate": params.values, "ci_low": lo.values,
                        "ci_high": hi.values, "p": np.asarray(p)})
    if kind == "logit":
        out[["estimate", "ci_low", "ci_high"]] = np.exp(out[["estimate", "ci_low", "ci_high"]])
        out["scale"] = "odds ratio"
    else:
        out["scale"] = "points"
    out = out[out["term"] != "const"].copy()
    out["label"] = out["term"].map(PRED_LABELS)
    out["method"] = method
    out["n"] = n
    out["outcome"] = outcome
    return out


def complete_case_model(d: pd.DataFrame, outcome: str, kind: str):
    df = add_model_terms(d)[PRED_NAMES + [outcome]].dropna()
    res = _fit(df, outcome, kind)
    tidy = _tidy(res.params, res.bse, None, kind, "complete case", len(df), outcome)
    return tidy, res, df


def vif_table(d: pd.DataFrame) -> pd.DataFrame:
    df = add_model_terms(d)[PRED_NAMES].dropna()
    X = sm.add_constant(df.astype(float))
    return pd.DataFrame({"term": PRED_NAMES, "label": [PRED_LABELS[p] for p in PRED_NAMES],
                         "vif": [variance_inflation_factor(X.values, i + 1) for i in range(len(PRED_NAMES))]})


def mice_models(d: pd.DataFrame, n_imputations: int = 20, burn_in: int = 10, seed: int = 20200505):
    """Multiple imputation (statsmodels MICEData, predictive mean matching) with Rubin's rules.

    The same pre-specified predictor set is fitted in every imputed data set.
    No variable selection is performed. SF-12 models use only rows whose 6 month
    SF-12 outcome was observed in the original file, as in the paper.
    """
    np.random.seed(seed)
    imp_df = d[IMPUTATION_COLUMNS].astype(float).reset_index(drop=True).copy()
    observed_pcs = imp_df["t2pcs12"].notna().values
    observed_mcs = imp_df["t2mcs12"].notna().values
    imp = MICEData(imp_df, k_pmm=10, rng=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp.update_all(burn_in)
        fits = {"rtw_yes": [], "t2pcs12": [], "t2mcs12": []}
        for _ in range(n_imputations):
            imp.update_all(1)
            cur = add_model_terms(imp.data.copy())
            for outcome, kind, mask in [("rtw_yes", "logit", None), ("t2pcs12", "ols", observed_pcs),
                                        ("t2mcs12", "ols", observed_mcs)]:
                sub = cur if mask is None else cur.loc[mask]
                res = _fit(sub, outcome, kind)
                fits[outcome].append((res.params, res.bse ** 2, len(sub)))
    out = []
    for outcome, kind in [("rtw_yes", "logit"), ("t2pcs12", "ols"), ("t2mcs12", "ols")]:
        P = pd.concat([f[0] for f in fits[outcome]], axis=1)
        V = pd.concat([f[1] for f in fits[outcome]], axis=1)
        m = P.shape[1]
        qbar = P.mean(axis=1)
        W = V.mean(axis=1)
        B = P.var(axis=1, ddof=1)
        T = W + (1 + 1 / m) * B
        r = (1 + 1 / m) * B / W
        df_rubin = (m - 1) * (1 + 1 / r) ** 2
        n = fits[outcome][0][2]
        res = _tidy(qbar, np.sqrt(T), df_rubin.clip(upper=1e6), kind,
                    f"multiple imputation (m={m})", n, outcome)
        res["fmi_approx"] = ((1 + 1 / m) * B / T).reindex(res["term"]).values
        out.append(res)
    return pd.concat(out, ignore_index=True)


def sf12_adjusted_rtw(d: pd.DataFrame) -> pd.DataFrame:
    """Age and sex adjusted OR of RTW per point of 6 month SF-12 (paper reports 1.08 / 1.06)."""
    rows = []
    for col, lab in [("t2pcs12", "PCS at 6 months"), ("t2mcs12", "MCS at 6 months")]:
        df = d[["rtw_yes", col, "age", "female"]].dropna()
        res = smf.glm(f"rtw_yes ~ {col} + age + female", data=df, family=sm.families.Binomial()).fit()
        b, se = res.params[col], res.bse[col]
        rows.append({"predictor": f"SF-12 {lab} (per point)", "n": len(df), "or": np.exp(b),
                     "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se),
                     "p": res.pvalues[col]})
    return pd.DataFrame(rows)


def exploratory_6mwd(d: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: is the change in 6MWD during CR associated with 6 month outcomes?

    Adjusted for age, sex and baseline 6MWD only. Restricted to the paired 6MWD subset.
    Not part of the paper's final models; interpret as descriptive association.
    """
    df = d.copy()
    df["d6mwd_50"] = (df["t16mwd"] - df["t06mwd"]) / 50
    df["mwd0_50"] = df["t06mwd"] / 50
    rows = []
    for outcome, fam, scale in [("rtw_yes", sm.families.Binomial(), "odds ratio"),
                                ("t2pcs12", sm.families.Gaussian(), "points"),
                                ("t2mcs12", sm.families.Gaussian(), "points")]:
        sub = df[[outcome, "d6mwd_50", "mwd0_50", "age", "female"]].dropna()
        res = smf.glm(f"{outcome} ~ d6mwd_50 + mwd0_50 + age + female", data=sub, family=fam).fit()
        b, se = res.params["d6mwd_50"], res.bse["d6mwd_50"]
        est, lo, hi = b, b - 1.96 * se, b + 1.96 * se
        if scale == "odds ratio":
            est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
        rows.append({"outcome": outcome, "term": "Change in 6MWD (per 50 m)", "n": len(sub),
                     "scale": scale, "estimate": est, "ci_low": lo, "ci_high": hi,
                     "p": res.pvalues["d6mwd_50"]})
    return pd.DataFrame(rows)
