"""Feature engineering for the live first-down probability model.

Everything comes from the project's ``data/`` folder (no outside data).

Two feature sets:
  * Pre-snap features: one row per play, from plays.csv + games.csv.
  * Frame features:    one row per tracking frame, from the snap until the pocket phase
                       ends (pass thrown, sack, or QB starts to run).

Tracking coordinates are normalized so the offense always moves toward +x (right).
Angles follow the NGS convention: 0 deg points toward +y, increasing clockwise (90 deg = +x).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIELD_W, FIELD_H = 120.0, 53.3
FPS = 10
K = ["playId", "frameId"]

SNAP_EVENTS = ["ball_snap"]
SNAP_FALLBACK = ["autoevent_ballsnap"]
# The first of these to happen ends the "pocket phase" we model.
END_EVENTS = ["pass_forward", "qb_sack", "qb_strip_sack", "run"]
END_FALLBACK = ["autoevent_passforward", "autoevent_passinterrupted"]

PRE_SNAP_NUMERIC = [
    "down", "yardsToGo", "yards_to_goal", "goal_to_go", "quarter", "half_seconds_left",
    "score_diff", "defendersInBox", "off_RB", "off_TE", "off_WR", "def_DL", "def_LB", "def_DB",
]
PRE_SNAP_CATEGORICAL = ["offenseFormation"]
PRE_SNAP_FEATURES = PRE_SNAP_NUMERIC + PRE_SNAP_CATEGORICAL

FRAME_FEATURES = [
    # time / phase
    "t", "play_action_shown", "is_end_frame", "ended_pass", "ended_sack", "ended_scramble",
    # quarterback
    "qb_depth", "qb_lateral", "qb_s",
    # pass rush
    "n_rushers", "rush_min_dist", "rush_n_within_2", "rush_n_within_3_5", "nearest_rusher_closing",
    "nearest_rusher_block_dist", "rush_time_to_qb", "n_free_rushers", "free_rusher_min_dist",
    "def_min_dist_qb",
    # protection / pocket
    "n_blockers", "pocket_area", "block_mean_retreat", "block_max_retreat",
    # receivers vs coverage
    "n_routes", "rec_max_sep", "rec_n_open", "rec_n_past_sticks", "rec_max_sep_past_sticks",
    "rec_max_depth_vs_sticks", "most_open_depth_vs_sticks",
]


# ---------------------------------------------------------------------------
# Labels + pre-snap features
# ---------------------------------------------------------------------------
def _count_personnel(series: pd.Series, pos: str) -> pd.Series:
    """'1 RB, 1 TE, 3 WR' -> count for one position code (0 if absent)."""
    return series.fillna("").str.extract(rf"(\d+)\s*{pos}\b")[0].astype(float).fillna(0.0)


def load_plays(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Plays with the `converted` label and pre-snap features.

    Dropped:
      * plays nullified by a penalty ("No Play"), since their result didn't count;
      * two-point tries (down == 0), which aren't normal downs.

    converted = 1 when the offense completes a pass or scrambles AND either gains at least
    `yardsToGo` before penalties or scores a touchdown. Interceptions/sacks returned for a
    defensive touchdown therefore stay 0.
    """
    plays = pd.read_csv(data_dir / "plays.csv")
    games = pd.read_csv(data_dir / "games.csv")

    desc = plays["playDescription"].fillna("")
    keep = ~desc.str.contains("No Play", regex=False) & (plays["down"] > 0)
    plays = plays.loc[keep].copy()
    desc = desc.loc[keep]

    offense_play = plays["passResult"].isin(["C", "R"])
    gained = plays["prePenaltyPlayResult"] >= plays["yardsToGo"]
    touchdown = desc.str.contains("TOUCHDOWN", regex=False)
    plays["converted"] = (offense_play & (gained | touchdown)).astype(int)

    plays = plays.merge(games[["gameId", "week", "homeTeamAbbr"]], on="gameId", how="left")
    is_home = plays["possessionTeam"] == plays["homeTeamAbbr"]
    home_margin = plays["preSnapHomeScore"] - plays["preSnapVisitorScore"]
    plays["score_diff"] = np.where(is_home, home_margin, -home_margin)

    # yardlineSide is empty at midfield; otherwise it's the half of the field the ball is in.
    own_half = plays["yardlineSide"] == plays["possessionTeam"]
    plays["yards_to_goal"] = np.where(
        plays["yardlineSide"].isna(), 50,
        np.where(own_half, 100 - plays["yardlineNumber"], plays["yardlineNumber"]),
    ).astype(float)
    plays["goal_to_go"] = (plays["yardsToGo"] >= plays["yards_to_goal"]).astype(int)

    clock = plays["gameClock"].str.split(":", expand=True).astype(float)
    secs = clock[0] * 60 + clock[1]
    plays["half_seconds_left"] = secs + np.where(plays["quarter"].isin([1, 3]), 900, 0)

    for pos in ["RB", "TE", "WR"]:
        plays[f"off_{pos}"] = _count_personnel(plays["personnelO"], pos)
    for pos in ["DL", "LB", "DB"]:
        plays[f"def_{pos}"] = _count_personnel(plays["personnelD"], pos)

    plays["offenseFormation"] = plays["offenseFormation"].astype("category")
    return plays.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tracking frame features
# ---------------------------------------------------------------------------
def _first_event(ball: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    hit = ball[ball["event"].isin(names)].sort_values(K)
    return hit.groupby("playId")[["frameId", "event"]].first()


def _play_windows(ball: pd.DataFrame) -> pd.DataFrame:
    """Per play: snap frame, end-of-pocket frame and event, play-action frame."""
    snap = _first_event(ball, SNAP_EVENTS)["frameId"].combine_first(
        _first_event(ball, SNAP_FALLBACK)["frameId"])
    end = _first_event(ball, END_EVENTS).combine_first(_first_event(ball, END_FALLBACK))

    meta = pd.DataFrame({"snap_frame": snap})
    meta["end_frame"] = end["frameId"]
    meta["end_event"] = end["event"]
    meta["last_frame"] = ball.groupby("playId")["frameId"].max()
    meta["pa_frame"] = _first_event(ball, ["play_action"])["frameId"]
    meta["end_event"] = meta["end_event"].fillna("none")
    meta["end_frame"] = meta["end_frame"].fillna(meta["last_frame"])
    meta = meta[meta["end_frame"] >= meta["snap_frame"]]
    meta[["snap_frame", "end_frame"]] = meta[["snap_frame", "end_frame"]].astype(int)
    return meta


def _hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of a small 2-D point set (monotone chain + shoelace)."""
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def game_frame_features(game_id: int, plays_g: pd.DataFrame, roles_g: pd.DataFrame,
                        data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Frame-level features for every kept play in one game."""
    cols = ["playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "dir", "event"]
    trk = pd.read_csv(Path(data_dir) / "tracking" / f"tracking_{game_id}.csv", usecols=cols)
    trk = trk[trk["playId"].isin(plays_g["playId"])].copy()
    if trk.empty:
        return pd.DataFrame()

    # Normalize so the offense always moves toward +x.
    left = trk["playDirection"].eq("left").to_numpy()
    trk["x"] = np.where(left, FIELD_W - trk["x"], trk["x"])
    trk["y"] = np.where(left, FIELD_H - trk["y"], trk["y"])
    trk["dir"] = np.where(left, (trk["dir"] + 180) % 360, trk["dir"])

    ball = trk[trk["team"] == "football"]
    meta = _play_windows(ball)
    first_ball = ball.sort_values(K).groupby("playId")[["x", "y"]].first()
    meta = meta.join(first_ball.rename(columns={"x": "ball_x0", "y": "ball_y0"}))
    meta = meta.join(plays_g.set_index("playId")[["absoluteYardlineNumber", "yardsToGo"]], how="inner")

    # absoluteYardlineNumber is the line of scrimmage in raw tracking x; normalize it too.
    is_left = trk.groupby("playId")["playDirection"].first().reindex(meta.index).eq("left")
    los = np.where(is_left, FIELD_W - meta["absoluteYardlineNumber"], meta["absoluteYardlineNumber"])
    meta["los_x"] = pd.Series(los, index=meta.index).fillna(meta["ball_x0"])
    meta["fd_x"] = np.minimum(meta["los_x"] + meta["yardsToGo"], 110.0)

    # Frames in the modeled window [snap, end of pocket phase].
    frames = ball[K].merge(meta[["snap_frame", "end_frame"]], left_on="playId", right_index=True)
    frames = frames[(frames["frameId"] >= frames["snap_frame"]) & (frames["frameId"] <= frames["end_frame"])][K]

    ppl = trk[trk["team"] != "football"].copy()
    ppl["nflId"] = ppl["nflId"].astype(int)
    ppl = ppl.merge(roles_g, on=["playId", "nflId"], how="left").merge(frames, on=K)

    def role(name: str) -> pd.DataFrame:
        return ppl.loc[ppl["pff_role"] == name, K + ["nflId", "x", "y", "s", "dir"]]

    qb = (role("Pass").groupby(K, as_index=False).first()
          .rename(columns={"nflId": "qb_id", "x": "qb_x", "y": "qb_y", "s": "qb_s", "dir": "qb_dir"}))
    rush, block, route, cov = role("Pass Rush"), role("Pass Block"), role("Pass Route"), role("Coverage")
    defense = pd.concat([rush, cov], ignore_index=True)

    out = frames.merge(meta[["snap_frame", "end_frame", "end_event", "pa_frame", "los_x", "ball_x0"]],
                       left_on="playId", right_index=True)
    out["t"] = (out["frameId"] - out["snap_frame"]) / FPS
    at_end = out["frameId"] == out["end_frame"]
    out["is_end_frame"] = at_end.astype(int)
    out["ended_pass"] = (at_end & out["end_event"].isin(["pass_forward", "autoevent_passforward"])).astype(int)
    out["ended_sack"] = (at_end & out["end_event"].isin(["qb_sack", "qb_strip_sack"])).astype(int)
    out["ended_scramble"] = (at_end & out["end_event"].eq("run")).astype(int)
    out["play_action_shown"] = (out["frameId"] >= out["pa_frame"]).astype(int)  # NaN -> False
    out["n_frames_play"] = out.groupby("playId")["frameId"].transform("size")

    # --- Quarterback ---
    q = qb.merge(meta[["los_x", "ball_y0"]], left_on="playId", right_index=True)
    q["qb_depth"] = q["los_x"] - q["qb_x"]            # yards behind the line of scrimmage
    q["qb_lateral"] = (q["qb_y"] - q["ball_y0"]).abs()  # sideways drift from where the ball was snapped
    out = out.merge(q[K + ["qb_depth", "qb_lateral", "qb_s"]], on=K, how="left")

    # --- Pass rush ---
    r = rush.merge(qb[K + ["qb_x", "qb_y"]], on=K)
    dx, dy = r["qb_x"] - r["x"], r["qb_y"] - r["y"]
    r["dist"] = np.hypot(dx, dy)
    rad = np.deg2rad(r["dir"])
    # speed component pointed at the QB (yd/s); positive = closing in
    r["closing"] = r["s"] * (np.sin(rad) * dx + np.cos(rad) * dy) / r["dist"].clip(lower=0.1)

    rb = rush[K + ["nflId", "x", "y"]].merge(block[K + ["x", "y"]], on=K, suffixes=("", "_b"))
    rb["d"] = np.hypot(rb["x"] - rb["x_b"], rb["y"] - rb["y_b"])
    nearest_block = rb.groupby(K + ["nflId"])["d"].min().rename("block_dist").reset_index()
    r = r.merge(nearest_block, on=K + ["nflId"], how="left")
    r["block_dist"] = r["block_dist"].fillna(99.0)
    r["free"] = r["block_dist"] > 1.5  # no blocker within 1.5 yards
    r["free_dist"] = r["dist"].where(r["free"])
    r["w2"], r["w35"] = r["dist"] < 2.0, r["dist"] < 3.5
    r = r.sort_values(K + ["dist"])
    g = r.groupby(K)
    ragg = g.agg(n_rushers=("dist", "size"), rush_min_dist=("dist", "min"),
                 rush_n_within_2=("w2", "sum"), rush_n_within_3_5=("w35", "sum"),
                 n_free_rushers=("free", "sum"), free_rusher_min_dist=("free_dist", "min"))
    nearest = g[["closing", "block_dist"]].first().rename(
        columns={"closing": "nearest_rusher_closing", "block_dist": "nearest_rusher_block_dist"})
    ragg = ragg.join(nearest).reset_index()
    ragg["rush_time_to_qb"] = (ragg["rush_min_dist"] / ragg["nearest_rusher_closing"].clip(lower=0.5)).clip(upper=10)
    out = out.merge(ragg, on=K, how="left")

    dq = defense[K + ["x", "y"]].merge(qb[K + ["qb_x", "qb_y"]], on=K)
    dq["d"] = np.hypot(dq["x"] - dq["qb_x"], dq["y"] - dq["qb_y"])
    out = out.merge(dq.groupby(K)["d"].min().rename("def_min_dist_qb").reset_index(), on=K, how="left")

    # --- Protection / pocket ---
    pts = pd.concat([block[K + ["x", "y"]],
                     qb[K + ["qb_x", "qb_y"]].rename(columns={"qb_x": "x", "qb_y": "y"})],
                    ignore_index=True).sort_values(K)
    if len(pts):
        keys = pts[K].to_numpy()
        xy = pts[["x", "y"]].to_numpy()
        starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
        ends = np.r_[starts[1:], len(pts)]
        hull = pd.DataFrame(keys[starts], columns=K)
        hull["pocket_area"] = [_hull_area(xy[s:e]) for s, e in zip(starts, ends)]
        out = out.merge(hull, on=K, how="left")
    else:
        out["pocket_area"] = np.nan

    b = block.merge(meta[["snap_frame"]], left_on="playId", right_index=True)
    b_snap = b.loc[b["frameId"] == b["snap_frame"], ["playId", "nflId", "x"]].rename(columns={"x": "x_snap"})
    b = block.merge(b_snap, on=["playId", "nflId"], how="left")
    b["retreat"] = b["x_snap"] - b["x"]  # yards moved back toward own end zone since the snap
    out = out.merge(b.groupby(K).agg(n_blockers=("x", "size"), block_mean_retreat=("retreat", "mean"),
                                     block_max_retreat=("retreat", "max")).reset_index(), on=K, how="left")

    # --- Receivers vs coverage ---
    rd = route[K + ["nflId", "x", "y"]].merge(defense[K + ["x", "y"]], on=K, suffixes=("", "_d"))
    rd["d"] = np.hypot(rd["x"] - rd["x_d"], rd["y"] - rd["y_d"])
    sep = rd.groupby(K + ["nflId"])["d"].min().rename("sep").reset_index()
    rec = (route[K + ["nflId", "x"]].merge(sep, on=K + ["nflId"], how="left")
           .merge(meta[["fd_x"]], left_on="playId", right_index=True))
    rec["depth_vs_sticks"] = rec["x"] - rec["fd_x"]
    rec["past_sticks"] = rec["depth_vs_sticks"] >= 0
    rec["open"] = rec["sep"] >= 3.0
    rec["sep_past"] = rec["sep"].where(rec["depth_vs_sticks"] >= -1.0)
    recagg = rec.groupby(K).agg(n_routes=("sep", "size"), rec_max_sep=("sep", "max"), rec_n_open=("open", "sum"),
                                rec_n_past_sticks=("past_sticks", "sum"),
                                rec_max_sep_past_sticks=("sep_past", "max"),
                                rec_max_depth_vs_sticks=("depth_vs_sticks", "max"))
    most_open = (rec.sort_values(K + ["sep"], ascending=[True, True, False]).groupby(K)["depth_vs_sticks"]
                 .first().rename("most_open_depth_vs_sticks"))
    out = out.merge(recagg.join(most_open).reset_index(), on=K, how="left")

    # Sensible fills where "none present" has a natural value.
    out["free_rusher_min_dist"] = out["free_rusher_min_dist"].fillna(30.0)
    out["rec_max_sep_past_sticks"] = out["rec_max_sep_past_sticks"].fillna(0.0)
    for c in ["n_rushers", "rush_n_within_2", "rush_n_within_3_5", "n_free_rushers", "n_blockers",
              "n_routes", "rec_n_open", "rec_n_past_sticks"]:
        out[c] = out[c].fillna(0)

    out.insert(0, "gameId", game_id)
    return out.drop(columns=["snap_frame", "end_frame", "pa_frame"])


def build_frame_features(plays: pd.DataFrame, data_dir: Path = DATA_DIR,
                         cache: str | Path | None = None) -> pd.DataFrame:
    """Frame features for all games (~0.2 s per game). Cached to a pickle if `cache` is given."""
    if cache is not None and Path(cache).exists():
        return pd.read_pickle(cache)

    pff = pd.read_csv(Path(data_dir) / "pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    roles = {gid: g[["playId", "nflId", "pff_role"]] for gid, g in pff.groupby("gameId")}
    parts = []
    for gid, pg in plays.groupby("gameId"):
        parts.append(game_frame_features(gid, pg[["playId", "absoluteYardlineNumber", "yardsToGo"]],
                                         roles.get(gid, pff.iloc[:0, 1:]), data_dir))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df
