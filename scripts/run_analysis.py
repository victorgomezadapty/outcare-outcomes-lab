"""Run the full OutCaRe Outcomes Lab analysis offline.

Usage (from the project root):
    python scripts/run_analysis.py

Writes aggregate tables to tables/, figures to figures/, a machine readable
summary to results/summary.json and regenerates RESULTS.md.
Set OUTCARE_FAST=1 to use 5 imputations instead of 20 (for quick checks).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outcare_lab import analysis as A  # noqa: E402
from outcare_lab import data as D  # noqa: E402
from outcare_lab import figures as F  # noqa: E402
from outcare_lab.report import write_results_md  # noqa: E402

TABLES, FIGS, RES = ROOT / "tables", ROOT / "figures", ROOT / "results"


def save(df: pd.DataFrame, name: str, digits: int = 3):
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(digits)
    out.to_csv(TABLES / f"{name}.csv", index=False)
    return out


def main():
    for p in (TABLES, FIGS, RES):
        p.mkdir(exist_ok=True)
    D.verify_raw()
    raw = D.load_raw()
    assert raw.shape == (D.EXPECTED_ROWS, D.EXPECTED_COLS), raw.shape
    ident = D.identifier_like_columns(raw.columns)
    assert not ident, f"identifier-like columns found: {ident}"
    d = D.prepare(raw)
    m_imp = 5 if os.environ.get("OUTCARE_FAST") == "1" else 20

    base = A.baseline_table(d)
    miss = A.missingness_table(d, [c for c in raw.columns if raw[c].isna().any()] + ["rtw"])
    cont = A.continuous_change(d)
    binc = A.binary_change(d)
    fu = A.followup_table(d)
    rtwf = A.rtw_by_factor(d)
    sel = A.selection_6mwd(d)
    sf12rtw = A.sf12_adjusted_rtw(d)
    vif = A.vif_table(d)
    expl = A.exploratory_6mwd(d)
    cc = []
    for outcome, kind in [("rtw_yes", "logit"), ("t2pcs12", "ols"), ("t2mcs12", "ols")]:
        t, _, _ = A.complete_case_model(d, outcome, kind)
        cc.append(t)
    cc = pd.concat(cc, ignore_index=True)
    mi = A.mice_models(d, n_imputations=m_imp)
    models = pd.concat([cc, mi], ignore_index=True)

    n = len(d)
    rtw_k = int(d["rtw_yes"].sum())
    checks = pd.DataFrame([
        {"check": "Follow-up patients (n)", "paper": "1262", "this_analysis": str(n)},
        {"check": "Age, mean (SD)", "paper": "54 (7)", "this_analysis": f"{d.age.mean():.1f} ({d.age.std():.1f})"},
        {"check": "Age range in file", "paper": "up to 65 (inclusion)", "this_analysis": f"{d.age.min()} to {d.age.max()}"},
        {"check": "Male, n (%)", "paper": "77%", "this_analysis": f"{int((d.sex == 1).sum())} ({100 * (d.sex == 1).mean():.1f}%)"},
        {"check": "Returned to work, n (%)", "paper": "864 (68.5%)", "this_analysis": f"{rtw_k} ({100 * rtw_k / n:.1f}%)"},
        {"check": "Retired at 6 months", "paper": "67 (5.3%)", "this_analysis": f"{int(d.t2retired_yes.sum())} ({100 * d.t2retired_yes.mean():.1f}%)"},
        {"check": "Applied for pension at 6 months", "paper": "79 (6.3%)", "this_analysis": f"{int(d.t2pension_yes.sum())} ({100 * d.t2pension_yes.mean():.1f}%)"},
        {"check": "Unemployed at 6 months", "paper": "89 (7.1%)", "this_analysis": f"{int(d.t2unemployed_yes.sum())} ({100 * d.t2unemployed_yes.mean():.1f}%)"},
        {"check": "Sick leave at 6 months", "paper": "190 (15.1%)", "this_analysis": f"{int(d.t2sick_yes.sum())} ({100 * d.t2sick_yes.mean():.1f}%)"},
        {"check": "Readmission at 6 months", "paper": "194 (15.4%)", "this_analysis": f"{int(d.t2hospital_yes.sum())} ({100 * d.t2hospital_yes.sum() / d.t2hospital_yes.notna().sum():.1f}% of {int(d.t2hospital_yes.notna().sum())})"},
        {"check": "Health moderately/greatly improved", "paper": "935 (74.7%)", "this_analysis": f"{int(d.t2health_yes.sum())} ({100 * d.t2health_yes.mean():.1f}% of {int(d.t2health_yes.notna().sum())})"},
        {"check": "Capacity enhanced", "paper": "876 (69.9%)", "this_analysis": f"{int(d.t2capacity_yes.sum())} ({100 * d.t2capacity_yes.mean():.1f}% of {int(d.t2capacity_yes.notna().sum())})"},
        {"check": "SF-12 at 6 months available", "paper": "1181 (94%)", "this_analysis": f"{int(d.t2pcs12.notna().sum())}"},
        {"check": "PCS / MCS at 6 months, mean (SD)", "paper": "45.5 (10) / 48.9 (11)", "this_analysis": f"{d.t2pcs12.mean():.1f} ({d.t2pcs12.std():.1f}) / {d.t2mcs12.mean():.1f} ({d.t2mcs12.std():.1f})"},
        {"check": "Smokers T0 to T1", "paper": "35% to 15%", "this_analysis": f"{binc.iloc[0].pct_t0:.1f}% to {binc.iloc[0].pct_t1:.1f}% (paired n={binc.iloc[0].n_paired})"},
        {"check": "Training load change", "paper": "+21 W", "this_analysis": f"{cont.iloc[1].mean_change:+.1f} W (paired n={cont.iloc[1].n_paired})"},
        {"check": "PCS / MCS change during CR", "paper": "+5.7 / +6.1", "this_analysis": f"{cont.iloc[9].mean_change:+.1f} / {cont.iloc[10].mean_change:+.1f} (paired n={cont.iloc[9].n_paired})"},
        {"check": "WHO-5 change during CR", "paper": "+18.3", "this_analysis": f"{cont.iloc[8].mean_change:+.1f} (paired n={cont.iloc[8].n_paired})"},
        {"check": "Negative occupational prognosis T0 to T1", "paper": "41% to 45%", "this_analysis": f"{binc.iloc[4].pct_t0:.1f}% to {binc.iloc[4].pct_t1:.1f}% (paired n={binc.iloc[4].n_paired})"},
        {"check": "Age/sex adjusted OR for RTW per PCS / MCS point at 6 months", "paper": "1.08 / 1.06", "this_analysis": f"{sf12rtw.iloc[0]['or']:.3f} / {sf12rtw.iloc[1]['or']:.3f}"},
        {"check": "OR pension desire T1 (paper final model)", "paper": "0.33 (0.22 to 0.50)", "this_analysis": "see model table"},
        {"check": "OR negative occupational prognosis T1 (paper final model)", "paper": "0.34 (0.24 to 0.48)", "this_analysis": "see model table"},
        {"check": "OR ACS (paper final model)", "paper": "0.64 (0.47 to 0.88)", "this_analysis": "see model table"},
    ])

    save(base, "t01_baseline_characteristics", 1)
    save(miss, "t02_missingness", 1)
    save(cont, "t03_change_continuous_T0_T1")
    save(binc, "t04_change_binary_T0_T1")
    save(fu, "t05_followup_6_months")
    save(rtwf, "t06_rtw_by_factor_unadjusted")
    save(models, "t07_models_rtw_sf12")
    save(vif, "t08_vif_complete_case")
    save(sf12rtw, "t09_rtw_vs_sf12_age_sex_adjusted")
    save(sel, "t10_6mwd_paired_subset_vs_rest")
    save(expl, "t11_exploratory_6mwd_change")
    save(checks, "t12_paper_sanity_checks")

    F.fig_missingness(miss, FIGS / "fig01_data_availability.png")
    F.fig_6mwd(d, FIGS / "fig02_6mwd_change.png")
    F.fig_training(d, FIGS / "fig03_training_load_change.png")
    F.fig_effect_sizes(cont, FIGS / "fig04_change_effect_sizes.png")
    F.fig_followup(fu, FIGS / "fig05_followup_status.png")
    F.fig_rtw_by_factor(rtwf, 100 * rtw_k / n, FIGS / "fig06_rtw_by_factor.png")
    F.fig_rtw_forest(models[models.outcome == "rtw_yes"], FIGS / "fig07_rtw_model_forest.png")
    F.fig_sf12_forest(models[models.outcome != "rtw_yes"], FIGS / "fig08_sf12_model_forest.png")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"doi": D.ZENODO_DOI, "license": "CC BY 4.0", "md5": D.md5sum(),
                   "bytes": D.EXPECTED_BYTES, "article_doi": D.ARTICLE_DOI},
        "n": n, "rtw_n": rtw_k, "rtw_pct": round(100 * rtw_k / n, 2),
        "n_imputations": m_imp,
        "complete_case_n": {o: int(cc[cc.outcome == o].n.iloc[0]) for o in cc.outcome.unique()},
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    write_results_md(ROOT / "RESULTS.md", d=d, base=base, cont=cont, binc=binc, fu=fu, rtwf=rtwf,
                     models=models, vif=vif, sf12rtw=sf12rtw, sel=sel, expl=expl, checks=checks,
                     miss=miss, m_imp=m_imp)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
