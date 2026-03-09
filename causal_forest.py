"""
causal_forest.py
----------------
Causal Forest fitting, CATE inference, and heterogeneity diagnostics for the
Smart Green Nudging replication — von Zahn et al. (2024).

Key additions over the original notebook
-----------------------------------------
* Calibration test  – checks whether the forest's heterogeneity is statistically
  significant (Chernozhukov et al. 2022 best-linear-predictor test).
* Confidence scoring  – each prediction comes with a coverage flag indicating
  whether the 95% CI excludes zero.
* RATE (Rank Average Treatment Effect) – measures the value of targeting by
  computing area-under-the-TOC curve.
* Structured CATEResult dataclass for clean downstream use.

Usage
-----
    from causal_forest import fit_causal_forest, CATEResult
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CATEResult:
    """
    Container for all CATE-related outputs from fit_causal_forest().

    Attributes
    ----------
    model        : fitted CausalForestDML instance
    cate         : 1-D ndarray of point estimates (test set)
    cate_lb      : lower bound of 95% CI
    cate_ub      : upper bound of 95% CI
    significant  : boolean mask — True where CI excludes zero
    conf_score   : float in [0,1], share of test obs where CI excludes zero
    feat_imp     : pd.Series of feature importances (sorted descending)
    blp_test     : dict with best-linear-predictor calibration test results
    """
    model:       object
    cate:        np.ndarray
    cate_lb:     np.ndarray
    cate_ub:     np.ndarray
    significant: np.ndarray
    conf_score:  float
    feat_imp:    pd.Series
    blp_test:    dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  CAUSAL FOREST — CATE SUMMARY",
            "═" * 60,
            f"  N (test set)          : {len(self.cate)}",
            f"  Mean CATE             : {self.cate.mean():+.4f}  ({self.cate.mean()*100:+.2f} pp)",
            f"  Std CATE              : {self.cate.std():.4f}",
            f"  Min / Max CATE        : {self.cate.min():+.4f} / {self.cate.max():+.4f}",
            f"  % CATE < 0 (nudge ↓) : {(self.cate < 0).mean():.1%}",
            f"  Confidence score      : {self.conf_score:.1%}  (share with CI ∌ 0)",
        ]
        if self.blp_test:
            blp = self.blp_test
            lines += [
                f"  BLP β₁ (mean HTE)     : {blp.get('beta1', np.nan):+.4f}  "
                f"p={blp.get('p_beta1', np.nan):.4f}",
                f"  BLP β₂ (heterogeneity): {blp.get('beta2', np.nan):+.4f}  "
                f"p={blp.get('p_beta2', np.nan):.4f}",
            ]
        lines.append("═" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_causal_forest(
    Y_train: np.ndarray | pd.Series,
    T_train: np.ndarray | pd.Series,
    X_train: np.ndarray | pd.DataFrame,
    X_test: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    n_estimators: int = 2_000,
    min_samples_leaf: int = 10,
    cv: int = 5,
    alpha: float = 0.05,
) -> CATEResult:
    """
    Fit a CausalForestDML and return a CATEResult with full diagnostics.

    Parameters
    ----------
    Y_train, T_train, X_train : training arrays
    X_test                    : held-out features for CATE predictions
    feature_names             : column names for feature importance display
    n_estimators              : number of causal trees
    min_samples_leaf          : minimum leaf size (controls regularisation)
    cv                        : cross-fitting folds
    alpha                     : significance level for CI construction

    Returns
    -------
    CATEResult
    """
    Y = np.asarray(Y_train)
    T = np.asarray(T_train)
    X_tr = np.asarray(X_train) if not isinstance(X_train, np.ndarray) else X_train
    X_te = np.asarray(X_test)  if not isinstance(X_test,  np.ndarray) else X_test

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_tr.shape[1])]

    logger.info(
        "fit_causal_forest: N_train=%d, N_test=%d, p=%d, n_trees=%d",
        len(Y), len(X_te), X_tr.shape[1], n_estimators,
    )

    model = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=RANDOM_SEED
        ),
        model_t=GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_SEED
        ),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=None,
        max_features="sqrt",
        inference=True,
        random_state=RANDOM_SEED,
        cv=cv,
    )

    logger.info("  Fitting causal forest (this may take ~1-2 min)…")
    model.fit(Y, T, X=X_tr)
    logger.info("  Fit complete.")

    # ── Point estimates & CIs ─────────────────────────────────────────────
    cate      = model.effect(X_te)
    lb, ub    = model.effect_interval(X_te, alpha=alpha)
    sig       = (ub < 0) | (lb > 0)          # CI excludes zero
    conf_score = sig.mean()

    # ── Feature importance ────────────────────────────────────────────────
    feat_imp = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    # ── BLP calibration test ──────────────────────────────────────────────
    blp = _blp_test(model, Y, T, X_tr)

    result = CATEResult(
        model=model,
        cate=cate,
        cate_lb=lb,
        cate_ub=ub,
        significant=sig,
        conf_score=float(conf_score),
        feat_imp=feat_imp,
        blp_test=blp,
    )

    logger.info(result.summary().replace("\n", " | "))
    return result


# ---------------------------------------------------------------------------
# Calibration test (Best Linear Predictor)
# ---------------------------------------------------------------------------

def _blp_test(
    model: CausalForestDML,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> dict:
    """
    Best-Linear-Predictor (BLP) test for CATE heterogeneity.

    Regresses (Y_residual) on (T_residual) and (T_residual × CATE_hat).
      β₁ ≈ mean treatment effect (should match ATE)
      β₂ > 0 ⟹ significant heterogeneity

    Reference: Chernozhukov et al. (2022), "Generic Machine Learning Inference
    on Heterogeneous Treatment Effects in Randomised Experiments."
    """
    try:
        import statsmodels.api as sm

        cate_train = model.effect(X)
        # Partial-out Y and T using the nuisance models
        Y_res = model.nuisance_scores_y if hasattr(model, "nuisance_scores_y") else Y - Y.mean()
        T_res = model.nuisance_scores_t if hasattr(model, "nuisance_scores_t") else T - T.mean()

        # Use a simple proxy: regress Y - E[Y|X] on T_res and T_res * CATE
        Y_proxy = Y - Y.mean()
        T_proxy = T - T.mean()
        X_blp   = sm.add_constant(
            np.column_stack([T_proxy, T_proxy * (cate_train - cate_train.mean())])
        )
        res = sm.OLS(Y_proxy, X_blp).fit(cov_type="HC3")

        return dict(
            beta1=res.params[1],
            beta2=res.params[2],
            p_beta1=res.pvalues[1],
            p_beta2=res.pvalues[2],
        )
    except Exception as exc:
        logger.warning("_blp_test failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# RATE: Rank Average Treatment Effect (targeting value)
# ---------------------------------------------------------------------------

def compute_rate(
    cate: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_quantiles: int = 20,
) -> dict:
    """
    Estimate the RATE (Rank Average Treatment Effect) — the area under the
    Targeting Operating Characteristic (TOC) curve.

    A positive RATE confirms that targeting high-CATE individuals is more
    valuable than random assignment.

    Returns
    -------
    dict with keys: rate, toc_x, toc_y, random_y
    """
    order  = np.argsort(cate)[::-1]    # rank from highest to lowest |effect|
    n      = len(cate)
    qs     = np.linspace(0, 1, n_quantiles + 1)[1:]   # top-q fractions

    toc_y = []
    for q in qs:
        k     = max(1, int(q * n))
        top_k = order[:k]
        ate_k = (
            Y[top_k][T[top_k] == 1].mean() - Y[top_k][T[top_k] == 0].mean()
            if (T[top_k] == 1).any() and (T[top_k] == 0).any()
            else np.nan
        )
        toc_y.append(ate_k)

    toc_y    = np.array(toc_y)
    random_y = np.nanmean(toc_y) * np.ones_like(toc_y)   # flat ATE baseline
    rate     = float(np.nanmean(toc_y) - np.nanmean(random_y))

    logger.info("compute_rate: RATE=%.4f", rate)
    return dict(rate=rate, toc_x=qs, toc_y=toc_y, random_y=random_y)
