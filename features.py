"""
features.py
-----------
Digital footprint feature engineering for the Smart Green Nudging replication.
Mirrors Table 1 / Appendix features from von Zahn et al. (2024).

Usage
-----
    from features import engineer_features, get_feature_cols
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct behavioural features from customer digital footprint columns.

    All transformations are safe: each block checks for the column's existence
    before operating, so the function works on any subset of the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially-processed dataframe containing some / all of the
        expected source columns (see below).

    Expected source columns (all optional)
    ----------------------------------------
    past_return_rate      : float  – historical return rate for this customer
    n_items_ordered       : int    – items in this order
    n_items_returned      : int    – items returned from this order (train only)
    order_value           : float  – gross order value (€)
    session_duration_sec  : float  – browsing session duration before purchase
    device_type           : str    – 'mobile' | 'desktop' | 'tablet'

    Returns
    -------
    pd.DataFrame with additional engineered columns appended.
    """
    df = df.copy()
    added: list[str] = []

    # ── Past return behaviour ──────────────────────────────────────────────
    if "past_return_rate" in df.columns:
        q75 = df["past_return_rate"].quantile(0.75)
        df["high_returner"] = (df["past_return_rate"] >= q75).astype(int)
        added.append("high_returner")

        # Quintile bins for finer-grained segmentation
        df["return_rate_quintile"] = pd.qcut(
            df["past_return_rate"], q=5, labels=False, duplicates="drop"
        )
        added.append("return_rate_quintile")

    # ── Bracketing behaviour ───────────────────────────────────────────────
    if "n_items_ordered" in df.columns and "n_items_returned" in df.columns:
        df["bracket_rate"] = (
            df["n_items_returned"] / df["n_items_ordered"].clip(lower=1)
        )
        added.append("bracket_rate")

    # ── Order value ────────────────────────────────────────────────────────
    if "order_value" in df.columns:
        df["log_order_value"] = np.log1p(df["order_value"])
        added.append("log_order_value")

        # High-value order flag (top tertile) — premium customer signal
        q67 = df["order_value"].quantile(0.67)
        df["high_value_order"] = (df["order_value"] >= q67).astype(int)
        added.append("high_value_order")

    # ── Session engagement ─────────────────────────────────────────────────
    if "session_duration_sec" in df.columns:
        df["log_session_duration"] = np.log1p(df["session_duration_sec"])
        added.append("log_session_duration")

    # ── Device type ────────────────────────────────────────────────────────
    if "device_type" in df.columns:
        dtype_lower = df["device_type"].str.lower()
        df["is_mobile"]  = (dtype_lower == "mobile").astype(int)
        df["is_tablet"]  = (dtype_lower == "tablet").astype(int)
        added += ["is_mobile", "is_tablet"]

    logger.info("engineer_features: added %d columns → %s", len(added), added)
    return df


def get_feature_cols(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Return the list of numeric feature columns, excluding treatment / outcome
    and any columns in `exclude`.

    Parameters
    ----------
    df      : pd.DataFrame – dataframe after engineer_features()
    exclude : list[str]    – additional column names to drop (default: [])

    Returns
    -------
    list[str] of numeric feature column names.
    """
    if exclude is None:
        exclude = []
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------

def check_covariate_balance(
    df: pd.DataFrame,
    treatment_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute standardised mean differences (SMD) between treatment arms for
    each feature.  SMD < 0.1 is considered well-balanced.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mean_control', 'mean_treated', 'smd']
    sorted by |SMD| descending.
    """
    rows = []
    for col in feature_cols:
        ctrl = df.loc[df[treatment_col] == 0, col].dropna()
        trt  = df.loc[df[treatment_col] == 1, col].dropna()
        pooled_sd = np.sqrt((ctrl.var() + trt.var()) / 2)
        smd = (trt.mean() - ctrl.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append(
            dict(
                feature=col,
                mean_control=ctrl.mean(),
                mean_treated=trt.mean(),
                smd=smd,
            )
        )
    result = pd.DataFrame(rows).sort_values("smd", key=abs, ascending=False)
    n_imbalanced = (result["smd"].abs() > 0.1).sum()
    if n_imbalanced:
        logger.warning(
            "check_covariate_balance: %d feature(s) have |SMD| > 0.1 — "
            "randomisation may be imperfect.",
            n_imbalanced,
        )
    return result.reset_index(drop=True)
