"""
plots.py
--------
All visualisations for the Smart Green Nudging replication.

Each function is self-contained: pass in data, get back a matplotlib Figure.
The notebook calls these as one-liners, keeping cells minimal.

Usage
-----
    from plots import (
        plot_descriptive_overview,
        plot_correlation_heatmap,
        plot_covariate_balance,
        plot_ate_forest,
        plot_permutation_null,
        plot_bandwidth_sensitivity,
        plot_cate_distribution,
        plot_feature_importance,
        plot_policy_curve,
        plot_cate_segments,
        plot_toc_curve,
    )
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
PALETTE = dict(control="#5B8DB8", nudge="#4CAF82", highlight="#E07B54")


# ---------------------------------------------------------------------------
# 1. Descriptive overview
# ---------------------------------------------------------------------------

def plot_descriptive_overview(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> plt.Figure:
    """Return rate and sample size by treatment arm."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [PALETTE["control"], PALETTE["nudge"]]

    df.groupby(treatment_col)[outcome_col].mean().plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="white"
    )
    axes[0].set_title("Return Rate by Treatment Arm")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Return Rate")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_xticklabels(["Control", "Nudge"], rotation=0)

    df[treatment_col].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color=colors, edgecolor="white"
    )
    axes[1].set_title("Sample Size by Treatment Arm")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(["Control", "Nudge"], rotation=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str,
    max_features: int = 12,
) -> plt.Figure:
    """Lower-triangle correlation heatmap of top features + outcome."""
    top_feats = feature_cols[: min(max_features, len(feature_cols))]
    corr = df[top_feats + [outcome_col]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, ax=ax, linewidths=0.5,
    )
    ax.set_title("Feature Correlation Matrix (incl. return outcome)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Covariate balance (SMD plot)
# ---------------------------------------------------------------------------

def plot_covariate_balance(balance_df: pd.DataFrame) -> plt.Figure:
    """
    Love plot of standardised mean differences (SMD).

    Parameters
    ----------
    balance_df : output of features.check_covariate_balance()
    """
    df = balance_df.copy().sort_values("smd")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

    colors = ["#E07B54" if abs(v) > 0.1 else "#5B8DB8" for v in df["smd"]]
    ax.barh(df["feature"], df["smd"], color=colors, edgecolor="white")
    ax.axvline(0,    color="black",     linestyle="-",  linewidth=0.8, alpha=0.5)
    ax.axvline(0.1,  color="#E07B54",   linestyle="--", linewidth=1,   alpha=0.7)
    ax.axvline(-0.1, color="#E07B54",   linestyle="--", linewidth=1,   alpha=0.7)
    ax.set_xlabel("Standardised Mean Difference (SMD)")
    ax.set_title("Covariate Balance — Love Plot\n(|SMD| > 0.1 flagged in orange)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. ATE forest plot
# ---------------------------------------------------------------------------

def plot_ate_forest(results_df: pd.DataFrame) -> plt.Figure:
    """
    Forest plot of ATE estimates with 95% CIs.

    Parameters
    ----------
    results_df : output of ATEResults.to_dataframe()
    """
    fig, ax = plt.subplots(figsize=(7, max(3, len(results_df) * 0.8)))
    colors = ["#5B8DB8", "#4CAF82", "#E07B54", "#9B59B6"]

    for i, row in results_df.iterrows():
        color = colors[i % len(colors)]
        ax.errorbar(
            row["ate"], i,
            xerr=[[row["ate"] - row["ci_lo"]], [row["ci_hi"] - row["ate"]]],
            fmt="o", color=color, markersize=9, capsize=5, linewidth=2,
        )
        # Star annotation
        from ate import _sig_stars
        stars = _sig_stars(row["pvalue"])
        if stars:
            ax.annotate(
                stars,
                xy=(row["ci_hi"], i),
                xytext=(5, 0), textcoords="offset points",
                va="center", color=color, fontsize=10,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["method"])
    ax.set_xlabel("ATE (change in return probability)")
    ax.set_title("ATE Estimates with 95% Confidence Intervals")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Permutation null distribution
# ---------------------------------------------------------------------------

def plot_permutation_null(perm_result: dict) -> plt.Figure:
    """
    Histogram of permuted ATEs vs. observed ATE.

    Parameters
    ----------
    perm_result : output of ate.permutation_test()
    """
    perm_ates    = perm_result["perm_ates"]
    observed_ate = perm_result["observed_ate"]
    p_value      = perm_result["p_value"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perm_ates, bins=50, color="#5B8DB8", edgecolor="white",
            alpha=0.8, label="Permuted ATE (null)")
    ax.axvline(observed_ate, color="#E07B54", linewidth=2,
               label=f"Observed ATE = {observed_ate:+.4f}")
    ax.axvline(-abs(observed_ate), color="#E07B54", linewidth=2,
               linestyle="--", alpha=0.6)
    ax.set_xlabel("ATE")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Permutation Test — Null Distribution\n"
        f"Empirical p-value = {p_value:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Bandwidth sensitivity
# ---------------------------------------------------------------------------

def plot_bandwidth_sensitivity(bw_df: pd.DataFrame) -> plt.Figure:
    """
    ATE + 95% CI across bandwidth levels.

    Parameters
    ----------
    bw_df : output of ate.bandwidth_sensitivity()
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(bw_df["bandwidth"], bw_df["ate"], "o-",
            color="#5B8DB8", linewidth=2, markersize=7, label="ATE")
    ax.fill_between(bw_df["bandwidth"], bw_df["ci_lo"], bw_df["ci_hi"],
                    alpha=0.2, color="#5B8DB8", label="95% CI")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Bandwidth (fraction of sample retained)")
    ax.set_ylabel("ATE")
    ax.set_title("Bandwidth Sensitivity — OLS + Controls ATE")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. CATE distribution
# ---------------------------------------------------------------------------

def plot_cate_distribution(cate_result) -> plt.Figure:
    """
    Histogram of CATEs + pie chart of beneficial vs. non-beneficial.

    Parameters
    ----------
    cate_result : CATEResult from causal_forest.fit_causal_forest()
    """
    cate = cate_result.cate
    pct_benefit = (cate < 0).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(cate, bins=50, color="#4CAF82", edgecolor="white", alpha=0.85)
    axes[0].axvline(cate.mean(), color="red", linestyle="--",
                    label=f"Mean CATE = {cate.mean():.4f}")
    axes[0].axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("CATE (treatment effect)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Individual CATEs (Test Set)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

    # Confidence-coloured overlay
    sig_cate  = cate[cate_result.significant]
    nsig_cate = cate[~cate_result.significant]
    axes[0].hist(sig_cate,  bins=30, color="darkgreen", alpha=0.4,
                 label=f"Significant (n={len(sig_cate)})")
    axes[0].hist(nsig_cate, bins=30, color="grey",      alpha=0.3,
                 label=f"Insignificant (n={len(nsig_cate)})")
    axes[0].legend(fontsize=8)

    axes[1].pie(
        [pct_benefit, 1 - pct_benefit],
        labels=[
            f"Nudge reduces\nreturns ({pct_benefit:.1%})",
            f"Nudge ineffective\nor backfires ({1-pct_benefit:.1%})",
        ],
        colors=["#4CAF82", "lightcoral"],
        startangle=90, autopct="%1.1f%%", pctdistance=0.75,
    )
    axes[1].set_title(
        f"Share Benefiting from Nudge\n"
        f"(Confidence score: {cate_result.conf_score:.1%})"
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(feat_imp: pd.Series, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of causal forest feature importances."""
    data = feat_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    data.plot(kind="barh", ax=ax, color="#5B8DB8", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Causal Forest — Top {top_n} Feature Importances\n(heterogeneity drivers)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Policy / profit curve
# ---------------------------------------------------------------------------

def plot_policy_curve(
    fractions: np.ndarray,
    profit_smart: np.ndarray,
    profit_univ: np.ndarray,
    best_smart_frac: float,
) -> plt.Figure:
    """Smart vs. universal nudging policy curve."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(fractions * 100, profit_smart, color="#4CAF82", linewidth=2.5,
            label="Smart targeting (CATE-ranked)")
    ax.plot(fractions * 100, profit_univ,  color="#5B8DB8", linewidth=2.5,
            linestyle="--", label="Universal nudging (ATE-based)")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(best_smart_frac * 100, color="#4CAF82", linestyle=":",
               linewidth=1.5, alpha=0.7,
               label=f"Optimal share = {best_smart_frac:.0%}")

    ax.set_xlabel("% Customers Targeted with Green Nudge")
    ax.set_ylabel("Profit Gain vs. No Nudging (€)")
    ax.set_title("Targeting Policy Curve — Profit Gain: Smart vs. Universal Nudging")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. CATE by segment (quartile bar chart)
# ---------------------------------------------------------------------------

def plot_cate_segments(seg_stats: pd.DataFrame) -> plt.Figure:
    """Mean CATE per customer quartile."""
    colors = ["#2e8b57", "#5aaa78", "#f4a56a", "#e07b54"]
    fig, ax = plt.subplots(figsize=(8, 4))
    seg_stats["mean_cate"].plot(kind="bar", ax=ax,
                                 color=colors, edgecolor="white")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Mean CATE by Customer Segment (Quartile)")
    ax.set_xlabel("CATE Quartile")
    ax.set_ylabel("Mean CATE")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_xticklabels(seg_stats.index, rotation=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11. TOC / RATE curve
# ---------------------------------------------------------------------------

def plot_toc_curve(rate_result: dict) -> plt.Figure:
    """
    Targeting Operating Characteristic (TOC) curve.

    Parameters
    ----------
    rate_result : output of causal_forest.compute_rate()
    """
    x  = rate_result["toc_x"] * 100
    y  = rate_result["toc_y"]
    ry = rate_result["random_y"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y,  color="#4CAF82", linewidth=2.5,
            label=f"TOC curve (RATE = {rate_result['rate']:+.4f})")
    ax.plot(x, ry, color="grey",    linewidth=1.5,
            linestyle="--", label="Random targeting (ATE baseline)")
    ax.fill_between(x, ry, y, alpha=0.15, color="#4CAF82")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Top-q% Customers Targeted")
    ax.set_ylabel("Average Treatment Effect in Top-q")
    ax.set_title("Targeting Operating Characteristic (TOC) Curve\n"
                 "Area above dashed = value of personalisation")
    ax.legend()
    fig.tight_layout()
    return fig
