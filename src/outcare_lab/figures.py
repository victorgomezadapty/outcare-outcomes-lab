"""Figures (aggregate only) for the OutCaRe Outcomes Lab."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
SOURCE = "Data: OutCaRe registry, Salzwedel et al. 2020, Zenodo doi:10.5281/zenodo.3605618 (CC BY 4.0)"


def _foot(fig):
    fig.text(0.01, 0.005, SOURCE, fontsize=6, color=GREY, ha="left", va="bottom")


def fig_missingness(miss: pd.DataFrame, path):
    m = miss.sort_values("n_available")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(m["variable"], m["n_available"], color=BLUE)
    ax.axvline(1262, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Patients with a recorded value (of 1262)")
    ax.set_title("Data availability per analysis variable")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def _paired_hist(d, a, b, unit, title, path, bins):
    both = d[[a, b]].dropna()
    diff = both[b] - both[a]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    ax = axes[0]
    rng = (min(both.min()), max(both.max()))
    ax.hist(both[a], bins=bins, range=rng, alpha=0.6, color=GREY, label="Admission (T0)")
    ax.hist(both[b], bins=bins, range=rng, alpha=0.6, color=BLUE, label="Discharge (T1)")
    ax.set_xlabel(unit)
    ax.set_ylabel("Patients")
    ax.legend(frameon=False)
    ax.set_title(f"Distribution, n = {len(both)} with both values")
    ax = axes[1]
    ax.hist(diff, bins=bins, color=ORANGE)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(diff.mean(), color=BLUE, ls="--", lw=1, label=f"mean {diff.mean():+.1f}")
    ax.set_xlabel(f"Change T1 minus T0 ({unit.split('(')[-1].rstrip(')')})")
    ax.legend(frameon=False)
    ax.set_title("Within-patient change")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def fig_6mwd(d, path):
    _paired_hist(d, "t06mwd", "t16mwd", "6-minute walking distance (m)",
                 "6MWD at admission and discharge of cardiac rehabilitation", path, 30)


def fig_training(d, path):
    _paired_hist(d, "t0training", "t1training", "Endurance training load (W)",
                 "Ergometer endurance training load during cardiac rehabilitation", path, 25)


def fig_effect_sizes(change: pd.DataFrame, path):
    c = change.copy()
    # orient so that positive = improvement
    sign = np.where(c["better_direction"] == "lower", -1, 1)
    c["ses_oriented"] = c["ses"] * sign
    c["lo"] = np.minimum(c["ci_low"] * sign, c["ci_high"] * sign) / c["sd_t0"]
    c["hi"] = np.maximum(c["ci_low"] * sign, c["ci_high"] * sign) / c["sd_t0"]
    c = c.sort_values("ses_oriented")
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(c))
    ax.errorbar(c["ses_oriented"], y, xerr=[c["ses_oriented"] - c["lo"], c["hi"] - c["ses_oriented"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=2)
    ax.set_yticks(y, [f"{m} (n={n})" for m, n in zip(c["measure"], c["n_paired"])])
    for v in (0.2, 0.5, 0.8):
        ax.axvline(v, color=GREY, ls=":", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardised effect size (mean change / SD at T0)\noriented so that right = improvement")
    ax.set_title("Change from CR admission to discharge\n(paired cases per measure)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def fig_rtw_by_factor(tab: pd.DataFrame, overall: float, path):
    t = tab.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(t))
    ax.errorbar(t["rtw_pct"], y, xerr=[t["rtw_pct"] - t["ci_low"], t["ci_high"] - t["rtw_pct"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=2)
    ax.set_yticks(y, [f"{f}: {l} (n={n})" for f, l, n in zip(t["factor"], t["level"], t["n"])], fontsize=7)
    ax.axvline(overall, color=ORANGE, ls="--", lw=1, label=f"overall {overall:.1f}%")
    ax.set_xlabel("Returned to work at 6 months, % (95% Wilson CI)")
    ax.set_title("Unadjusted return to work by patient characteristic")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def _forest(ax, models: pd.DataFrame, ref: float, log: bool):
    methods = list(models["method"].unique())
    labels = list(dict.fromkeys(models["label"]))
    y = np.arange(len(labels))[::-1]
    offs = np.linspace(-0.18, 0.18, len(methods))
    colors = [BLUE, ORANGE, GREY]
    for i, meth in enumerate(methods):
        sub = models[models["method"] == meth].set_index("label").reindex(labels)
        n = int(sub["n"].iloc[0])
        ax.errorbar(sub["estimate"], y + offs[i],
                    xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
                    fmt="o", ms=3.5, color=colors[i], ecolor=colors[i], elinewidth=1, capsize=0,
                    label=f"{meth}, n={n}")
    ax.axvline(ref, color="k", lw=0.8)
    if log:
        ax.set_xscale("log")
        ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
        lo, hi = models["ci_low"].min(), models["ci_high"].max()
        ticks = [t for t in ticks if lo * 0.8 <= t <= hi * 1.25]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y, labels, fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def fig_rtw_forest(models: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _forest(ax, models, 1.0, True)
    ax.set_xlabel("Adjusted odds ratio for return to work at 6 months (95% CI, log scale)")
    ax.set_title("Return to work at 6 months: pre-specified model\n(adjusted associations, not causal effects)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def fig_sf12_forest(models: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, (outcome, title) in zip(axes, [("t2pcs12", "SF-12 PCS at 6 months"),
                                           ("t2mcs12", "SF-12 MCS at 6 months")]):
        _forest(ax, models[models["outcome"] == outcome], 0.0, False)
        ax.set_title(title)
        ax.set_xlabel("Adjusted difference in points (95% CI)")
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("SF-12 at 6 months: pre-specified linear models (adjusted associations, not causal effects)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)


def fig_followup(fu: pd.DataFrame, path):
    t = fu.dropna(subset=["pct"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(t))
    ax.barh(y, t["pct"], color=BLUE, alpha=0.85)
    ax.errorbar(t["pct"], y, xerr=[t["pct"] - t["ci_low"], t["ci_high"] - t["pct"]], fmt="none",
                ecolor="k", capsize=2, lw=0.8)
    ax.set_yticks(y, [f"{o} (n={n})" for o, n in zip(t["outcome_6_months"], t["n_available"])], fontsize=7)
    ax.set_xlabel("% of follow-up responders (95% Wilson CI)")
    ax.set_title("Status 6 months after cardiac rehabilitation")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _foot(fig)
    fig.savefig(path)
    plt.close(fig)
