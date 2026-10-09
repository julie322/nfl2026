"""Training + evaluation for the live first-down probability model.

Two models, both gradient-boosted trees (scikit-learn HistGradientBoostingClassifier):
  * pre-snap model:  P(converted | situation)              one row per play
  * live model:      P(converted | situation, frame t)     one row per tracking frame

Validation is grouped by game (5 folds) so frames from one play never sit in both train and
test. Every play therefore gets honest out-of-fold (OOF) predictions, which are what the
viewer and the leaderboards use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from features import FRAME_FEATURES, PRE_SNAP_FEATURES

LABEL = "converted"
PLAY_KEY = ["gameId", "playId"]

PRE_SNAP_PARAMS = dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=8, min_samples_leaf=60,
                       l2_regularization=1.0)
LIVE_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=400,
                   l2_regularization=1.0)


def make_model(params: dict) -> HistGradientBoostingClassifier:
    # Early stopping is off on purpose: its internal split is random by row, which would put
    # frames of the same play on both sides and stop too late.
    return HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=False,
                                          random_state=0, **params)


def build_frame_table(plays: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Join frame features with play-level label and pre-snap features."""
    cols = PLAY_KEY + [LABEL, "possessionTeam", "defensiveTeam", "passResult"] + PRE_SNAP_FEATURES
    df = frames.merge(plays[cols], on=PLAY_KEY, how="inner")
    # Each play counts once in training regardless of how many frames it has.
    df["w"] = 1.0 / df.groupby(PLAY_KEY)["frameId"].transform("size")
    return df.sort_values(PLAY_KEY + ["frameId"]).reset_index(drop=True)


# The live model sees the pre-snap model's opinion (as a logit) plus a few raw situation
# columns it can interact with frame features (e.g. receiver depth matters more on 3rd & long).
LIVE_FEATURES = ["presnap_logit", "down", "yardsToGo", "yards_to_goal"] + FRAME_FEATURES


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _presnap_oof(plays: pd.DataFrame, n_splits: int, params: dict) -> np.ndarray:
    """Out-of-fold pre-snap predictions within `plays` (grouped by game)."""
    out = np.full(len(plays), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(plays, groups=plays["gameId"]):
        m = make_model(params).fit(plays.iloc[tr][PRE_SNAP_FEATURES], plays.iloc[tr][LABEL])
        out[te] = m.predict_proba(plays.iloc[te][PRE_SNAP_FEATURES])[:, 1]
    return out


def _attach_presnap(df: pd.DataFrame, plays: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    s = plays[PLAY_KEY].assign(presnap_logit=_logit(p))
    return df.drop(columns=["presnap_logit"], errors="ignore").merge(s, on=PLAY_KEY, how="left")


def cross_validate(plays: pd.DataFrame, df: pd.DataFrame, n_splits: int = 5,
                   pre_params: dict = PRE_SNAP_PARAMS, live_params: dict = LIVE_PARAMS):
    """Returns (plays with OOF `p_presnap`, frames with OOF `p_live` and `p_presnap`).

    Nested so nothing leaks: inside each outer fold, the live model's `presnap_logit` feature
    for training games comes from an inner out-of-fold pre-snap model, and for test games from
    a pre-snap model fit on the outer training games only.
    """
    plays = plays[plays.set_index(PLAY_KEY).index.isin(df.set_index(PLAY_KEY).index.unique())]
    plays = plays.reset_index(drop=True).copy()
    plays["p_presnap"] = np.nan
    preds = []

    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(plays, groups=plays["gameId"]):
        tr_p, te_p = plays.iloc[tr_idx], plays.iloc[te_idx]

        pre = make_model(pre_params).fit(tr_p[PRE_SNAP_FEATURES], tr_p[LABEL])
        p_test = pre.predict_proba(te_p[PRE_SNAP_FEATURES])[:, 1]
        plays.loc[te_idx, "p_presnap"] = p_test

        tr = _attach_presnap(df[df["gameId"].isin(tr_p["gameId"])], tr_p, _presnap_oof(tr_p, n_splits, pre_params))
        te = _attach_presnap(df[df["gameId"].isin(te_p["gameId"])], te_p, p_test)

        live = make_model(live_params).fit(tr[LIVE_FEATURES], tr[LABEL], sample_weight=tr["w"])
        preds.append(te.assign(p_live=live.predict_proba(te[LIVE_FEATURES])[:, 1]))

    out = pd.concat(preds, ignore_index=True).merge(plays[PLAY_KEY + ["p_presnap"]], on=PLAY_KEY, how="left")
    return plays, out.sort_values(PLAY_KEY + ["frameId"]).reset_index(drop=True)


def fit_full(plays: pd.DataFrame, df: pd.DataFrame, n_splits: int = 5):
    """Final models trained on all data (for scoring new plays)."""
    pre = make_model(PRE_SNAP_PARAMS).fit(plays[PRE_SNAP_FEATURES], plays[LABEL])
    tr = _attach_presnap(df, plays, _presnap_oof(plays, n_splits, PRE_SNAP_PARAMS))
    live = make_model(LIVE_PARAMS).fit(tr[LIVE_FEATURES], tr[LABEL], sample_weight=tr["w"])
    return pre, live


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def scores(y, p, w=None) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p, sample_weight=w),
        "brier": brier_score_loss(y, p, sample_weight=w),
        "auc": roc_auc_score(y, p, sample_weight=w),
        "n": len(y),
    }


def summarize_plays(df: pd.DataFrame) -> pd.DataFrame:
    """One row per play: probability at the snap, at the end of the pocket phase, and the change."""
    g = df.groupby(PLAY_KEY, sort=False)
    out = g.agg(converted=(LABEL, "first"), possessionTeam=("possessionTeam", "first"),
                defensiveTeam=("defensiveTeam", "first"), passResult=("passResult", "first"),
                end_event=("end_event", "first"), p_presnap=("p_presnap", "first"),
                p_snap=("p_live", "first"), p_end=("p_live", "last"), time_to_end=("t", "max"),
                min_rush_dist=("rush_min_dist", "min"))
    out["delta"] = out["p_end"] - out["p_snap"]
    return out.reset_index()


def evaluation_table(plays: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    summ = summarize_plays(df)
    y = summ[LABEL]
    rows = {
        "Base rate (constant)": scores(y, np.full(len(y), plays[LABEL].mean())),
        "Pre-snap model": scores(y, summ["p_presnap"]),
        "Live model @ snap": scores(y, summ["p_snap"]),
        "Live model @ end of pocket": scores(y, summ["p_end"]),
        "Live model, all frames (play-weighted)": scores(df[LABEL], df["p_live"], df["w"]),
    }
    return pd.DataFrame(rows).T


def by_time_table(df: pd.DataFrame, bins=(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0)) -> pd.DataFrame:
    """Accuracy of pre-snap vs live model at each moment after the snap."""
    d = df.assign(t_bin=pd.cut(df["t"], bins=list(bins), right=False))
    rows = []
    for b, g in d.groupby("t_bin", observed=True):
        if g[LABEL].nunique() < 2:
            continue
        pre, live = scores(g[LABEL], g["p_presnap"]), scores(g[LABEL], g["p_live"])
        rows.append({"seconds_after_snap": str(b), "frames": len(g), "plays": g[PLAY_KEY].drop_duplicates().shape[0],
                     "presnap_log_loss": pre["log_loss"], "live_log_loss": live["log_loss"],
                     "presnap_auc": pre["auc"], "live_auc": live["auc"]})
    return pd.DataFrame(rows)


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
