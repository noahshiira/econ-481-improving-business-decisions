"""
ate.py
------
Average Treatment Effect (ATE) estimation for the Smart Green Nudging
replication — von Zahn et al. (2024).

Estimators
----------
1. OLS (naive, no controls)
2. OLS with covariate adjustment  (HC3 robust SEs)
3. Augmented IPW / Doubly-Robust  (AIPW — new, more efficient than plain IPW)
4. Plain Horvitz-Thompson IPW     (kept for comparison)
5. Permutation / placebo tests    (robustness)
6. Bandwidth sensitivity sweep    (robustness)

Usage
-----
    from ate import run_ate_suite, permutation_test, bandwidth_sensitivity
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ATEResults:
    """Holds ATE point estimates, CIs, p-values for all methods."""
    method:  list[str]  = field(default_factory=list)
    ate:     list[float] = field(default_factory=list)
    ci_lo:   list[float] = field(default_factory=list)
    ci_hi:   list[float] = field(default_factory=list)
    pvalue:  list[float] = field(default_factory=list)
    se:      list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            dict(
                method=self.method,
                ate=self.ate,
                ci_lo=self.ci_lo,
                ci_hi=self.ci_hi,
                pvalue=self.pvalue,
                se=self.se,
            )
        )

    def summary(self) -> str:
        df = self.to_dataframe()
        lines = ["═" * 72, "  ATE ESTIMATION RESULTS", "═" * 72]
        for _, row in df.iterrows():
            sig = _sig_stars(row["pvalue"])
            lines.append(
                f"  {row['method']:<30}  "
                f"ATE={row['ate']:+.4f}  "
                f"95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
                f"p={row['pvalue']:.4f} {sig}"
            )
        lines.append("═" * 72)
        lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
        return "\n".join(lines)


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


# ---------------------------------------------------------------------------
# Individual estimators
# ---------------------------------------------------------------------------

def _ols_naive(Y: pd.Series, T: pd.Series) -> dict:
    res = sm.OLS(Y, sm.add_constant(T)).fit(cov_type="HC3")
    ci  = res.conf_int().loc[T.name]
    return dict(
        method="OLS (naive)",
        ate=res.params[T.name],
        ci_lo=ci[0], ci_hi=ci[1],
        pvalue=res.pvalues[T.name],
        se=res.bse[T.name],
    )


def _ols_controlled(Y: pd.Series, T: pd.Series, X: pd.DataFrame) -> dict:
    Xmat = sm.add_constant(pd.concat([T, X], axis=1))
    res  = sm.OLS(Y, Xmat).fit(cov_type="HC3")
    ci   = res.conf_int().loc[T.name]
    return dict(
        method="OLS + covariates",
        ate=res.params[T.name],
        ci_lo=ci[0], ci_hi=ci[1],
        pvalue=res.pvalues[T.name],
        se=res.bse[T.name],
    )


def _propensity_scores(
    X: pd.DataFrame, T: pd.Series, clip: tuple[float, float] = (0.05, 0.95)
) -> np.ndarray:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.fillna(0))
    ps     = LogisticRegression(max_iter=1_000, random_state=42)
    ps.fit(X_sc, T)
    scores = ps.predict_proba(X_sc)[:, 1]
    return np.clip(scores, *clip)


def _ipw(Y: np.ndarray, T: np.ndarray, ps: np.ndarray,
         n_boot: int = 500, seed: int = 42) -> dict:
    """Horvitz-Thompson IPW with bootstrap SE."""
    def _ht(y, t, p):
        return (t * y / p).mean() - ((1 - t) * y / (1 - p)).mean()

    ate = _ht(Y, T, ps)
    rng = np.random.default_rng(seed)
    boots = [
        _ht(Y[idx], T[idx], ps[idx])
        for idx in (
            rng.choice(len(Y), size=len(Y), replace=True)
            for _ in range(n_boot)
        )
    ]
    se    = np.std(boots)
    z     = ate / se if se > 0 else np.nan
    pval  = 2 * (1 - _norm_cdf(abs(z)))
    return dict(
        method="IPW (Horvitz-Thompson)",
        ate=ate,
        ci_lo=ate - 1.96 * se, ci_hi=ate + 1.96 * se,
        pvalue=pval, se=se,
    )


def _aipw(
    Y: pd.Series, T: pd.Series, X: pd.DataFrame,
    ps: np.ndarray, n_boot: int = 500, seed: int = 42
) -> dict:
    """
    Augmented IPW (doubly-robust estimator).
    Outcome models are fit via OLS separately for each arm, then combined
    with the IPW correction.  Consistent if *either* the propensity model
    or the outcome model is correctly specified.
    """
    t_arr = T.values
    y_arr = Y.values

    # Outcome models
    m1 = sm.OLS(
        y_arr[t_arr == 1],
        sm.add_constant(X.fillna(0).values[t_arr == 1]),
    ).fit()
    m0 = sm.OLS(
        y_arr[t_arr == 0],
        sm.add_constant(X.fillna(0).values[t_arr == 0]),
    ).fit()

    X_c = sm.add_constant(X.fillna(0).values)
    mu1 = m1.predict(X_c)
    mu0 = m0.predict(X_c)

    def _dr_ate(y, t, p, m1_hat, m0_hat):
        dr = (
            (t * (y - m1_hat) / p)
            - ((1 - t) * (y - m0_hat) / (1 - p))
            + m1_hat
            - m0_hat
        )
        return dr.mean()

    ate  = _dr_ate(y_arr, t_arr, ps, mu1, mu0)
    rng  = np.random.default_rng(seed)
    idx_all = np.arange(len(y_arr))
    boots = [
        _dr_ate(
            y_arr[idx], t_arr[idx], ps[idx], mu1[idx], mu0[idx]
        )
        for idx in (
            rng.choice(idx_all, size=len(idx_all), replace=True)
            for _ in range(n_boot)
        )
    ]
    se   = np.std(boots)
    z    = ate / se if se > 0 else np.nan
    pval = 2 * (1 - _norm_cdf(abs(z)))
    return dict(
        method="AIPW (doubly-robust)",
        ate=ate,
        ci_lo=ate - 1.96 * se, ci_hi=ate + 1.96 * se,
        pvalue=pval, se=se,
    )


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF (avoids scipy dependency)."""
    from math import erfc, sqrt
    return 1 - 0.5 * erfc(x / sqrt(2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ate_suite(
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    n_boot: int = 500,
    ps_clip: tuple[float, float] = (0.05, 0.95),
    seed: int = 42,
) -> ATEResults:
    """
    Run the full ATE estimation suite: OLS naive, OLS+controls, IPW, AIPW.

    Parameters
    ----------
    Y       : outcome series (binary 0/1)
    T       : treatment series (binary 0/1)
    X       : covariate dataframe (numeric, already engineered)
    n_boot  : bootstrap replicates for IPW / AIPW SE
    ps_clip : propensity score clipping bounds
    seed    : random seed

    Returns
    -------
    ATEResults dataclass with .summary() and .to_dataframe() methods.
    """
    logger.info("run_ate_suite: N=%d, treatment rate=%.3f", len(Y), T.mean())

    results = ATEResults()
    ps = _propensity_scores(X, T, clip=ps_clip)

    for estimator_fn, args in [
        (_ols_naive,      (Y, T)),
        (_ols_controlled, (Y, T, X)),
        (_ipw,            (Y.values, T.values, ps, n_boot, seed)),
        (_aipw,           (Y, T, X, ps, n_boot, seed)),
    ]:
        try:
            row = estimator_fn(*args)
            for k, v in row.items():
                getattr(results, k).append(v)
            logger.info("  %-30s ATE=%+.4f  p=%.4f", row["method"], row["ate"], row["pvalue"])
        except Exception as exc:
            logger.error("  %s failed: %s", estimator_fn.__name__, exc)

    return results


# ---------------------------------------------------------------------------
# Robustness: permutation / placebo test
# ---------------------------------------------------------------------------

def permutation_test(
    Y: pd.Series,
    T: pd.Series,
    X: pd.DataFrame,
    n_permutations: int = 1_000,
    seed: int = 42,
) -> dict:
    """
    Permutation test for the OLS+controls ATE.

    Randomly shuffles the treatment label n_permutations times and computes
    the OLS+controls ATE each time.  The empirical p-value is the fraction of
    permuted ATEs with |ATE| >= |observed ATE|.

    Returns
    -------
    dict with keys: observed_ate, perm_ates, p_value, null_mean, null_std
    """
    logger.info("permutation_test: running %d permutations...", n_permutations)

    Xmat = sm.add_constant(pd.concat([T, X], axis=1))
    observed_ate = (
        sm.OLS(Y, Xmat).fit(cov_type="HC3").params[T.name]
    )

    rng  = np.random.default_rng(seed)
    perm_ates = []
    T_vals = T.values.copy()

    for _ in range(n_permutations):
        T_perm  = pd.Series(rng.permutation(T_vals), index=T.index, name=T.name)
        Xmat_p  = sm.add_constant(pd.concat([T_perm, X], axis=1))
        perm_ates.append(
            sm.OLS(Y, Xmat_p).fit().params[T_perm.name]
        )

    perm_ates = np.array(perm_ates)
    p_value   = (np.abs(perm_ates) >= abs(observed_ate)).mean()

    logger.info(
        "permutation_test: observed ATE=%+.4f, permutation p=%.4f",
        observed_ate, p_value,
    )
    return dict(
        observed_ate=observed_ate,
        perm_ates=perm_ates,
        p_value=p_value,
        null_mean=perm_ates.mean(),
        null_std=perm_ates.std(),
    )


# ---------------------------------------------------------------------------
# Robustness: bandwidth / sample-restriction sensitivity
# ---------------------------------------------------------------------------

def bandwidth_sensitivity(
    df: pd.DataFrame,
    Y_col: str,
    T_col: str,
    feature_cols: list[str],
    score_col: str | None = None,
    bandwidths: list[float] | None = None,
) -> pd.DataFrame:
    """
    Re-estimate OLS+controls ATE across progressively restricted subsamples.

    If `score_col` is provided the sample is restricted to observations within
    ±bandwidth standard deviations of that score's mean (mimicking an RD
    bandwidth sweep).  If `score_col` is None, bandwidths represent the
    fraction of the sample retained (by random sub-sampling).

    Parameters
    ----------
    df           : full dataframe
    Y_col        : outcome column name
    T_col        : treatment column name
    feature_cols : covariate column names
    score_col    : continuous score to restrict on (optional)
    bandwidths   : list of bandwidth values to sweep

    Returns
    -------
    pd.DataFrame with columns [bandwidth, n, ate, ci_lo, ci_hi, pvalue]
    """
    if bandwidths is None:
        bandwidths = [1.0, 0.9, 0.75, 0.5, 0.35, 0.25, 0.15]

    rows = []
    for bw in bandwidths:
        if score_col is not None:
            mu  = df[score_col].mean()
            sd  = df[score_col].std()
            sub = df[(df[score_col] >= mu - bw * sd) & (df[score_col] <= mu + bw * sd)]
        else:
            sub = df.sample(frac=bw, random_state=42)

        if len(sub) < 50:
            logger.warning("bandwidth_sensitivity: only %d obs at bw=%.2f — skipping", len(sub), bw)
            continue

        Y = sub[Y_col]
        T = sub[T_col]
        X = sub[feature_cols].fillna(0)

        Xmat = sm.add_constant(pd.concat([T, X], axis=1))
        res  = sm.OLS(Y, Xmat).fit(cov_type="HC3")
        ci   = res.conf_int().loc[T_col]
        rows.append(
            dict(
                bandwidth=bw,
                n=len(sub),
                ate=res.params[T_col],
                ci_lo=ci[0],
                ci_hi=ci[1],
                pvalue=res.pvalues[T_col],
                sig=_sig_stars(res.pvalues[T_col]),
            )
        )

    result = pd.DataFrame(rows)
    logger.info("bandwidth_sensitivity: completed %d bandwidth levels", len(result))
    return result
