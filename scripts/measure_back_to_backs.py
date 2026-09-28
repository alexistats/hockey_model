"""How back-to-backs move goalie starts, and whether counting them predicts better.

    python scripts/measure_back_to_backs.py

Every regular-season team game is the first night of a back-to-back (the team
plays the next calendar day), the second (it played the day before), or neither.
For every goalie who spent a season on one team: his share of the team's starts,
and his start rate on each kind of night. The second-night rate by share is
`BACK_TO_BACK_SECOND` in hockey/serve/schedule.py; 2020-21, a 56-game season of
two-game series, is left out.

Then the check that earns it a place: the curve measured on the seasons before
the last one, scored on the last against who actually started - per game, and per
fantasy week's expected starts - with each goalie's actual season share, so the
comparison isolates what the kind of night adds.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from hockey.db import SessionLocal
from hockey.serve.schedule import start_chances

SKIP = {20202021}
BANDS = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01]
MIN_GOALIES = 20  # a band thinner than this is noise, not a knot


def team_games() -> pd.DataFrame:
    with SessionLocal() as s:
        games = pd.DataFrame(
            s.execute(
                text(
                    "SELECT nhl_game_id AS game_id, season, date, home_team_abbrev AS home, "
                    "away_team_abbrev AS away FROM nhl_games WHERE game_type = 2"
                )
            ).mappings()
        )
    games["date"] = pd.to_datetime(games["date"])
    tg = pd.concat(
        [
            games[["game_id", "season", "date", "home"]].rename(columns={"home": "team"}),
            games[["game_id", "season", "date", "away"]].rename(columns={"away": "team"}),
        ]
    ).sort_values(["team", "season", "date"])
    prev = tg.groupby(["team", "season"])["date"].shift(1)
    nxt = tg.groupby(["team", "season"])["date"].shift(-1)
    tg["kind"] = np.where(
        (tg["date"] - prev).dt.days.eq(1),
        "second",
        np.where((nxt - tg["date"]).dt.days.eq(1), "first", "rest"),
    )
    return tg


def goalie_seasons(tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with SessionLocal() as s:
        starts = pd.DataFrame(
            s.execute(
                text(
                    "SELECT gl.player_id, gl.game_id, gl.team_abbrev AS team "
                    "FROM goalie_game_logs gl JOIN nhl_games g ON g.nhl_game_id = gl.game_id "
                    "WHERE g.game_type = 2 AND gl.started"
                )
            ).mappings()
        )
    st = starts.merge(tg, on=["game_id", "team"])
    one_team = st.groupby(["player_id", "season"])["team"].nunique()
    keep = set(one_team[one_team == 1].index)
    st = st[[k in keep for k in zip(st.player_id, st.season, strict=True)]]
    rows = []
    for (pid, season, team), grp in st.groupby(["player_id", "season", "team"]):
        sched = tg[(tg.team == team) & (tg.season == season)]
        got = grp["kind"].value_counts()
        row = {"player_id": pid, "season": season, "team": team, "share": len(grp) / len(sched)}
        for k in ("first", "second", "rest"):
            row[f"n_{k}"] = int((sched.kind == k).sum())
            row[f"s_{k}"] = int(got.get(k, 0))
        rows.append(row)
    return pd.DataFrame(rows), st


def knots(g: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    t = (
        g.assign(band=pd.cut(g["share"], BANDS, right=False))
        .groupby("band", observed=True)
        .agg(
            goalies=("player_id", "size"),
            share=("share", "mean"),
            s_first=("s_first", "sum"),
            n_first=("n_first", "sum"),
            s_second=("s_second", "sum"),
            n_second=("n_second", "sum"),
            s_rest=("s_rest", "sum"),
            n_rest=("n_rest", "sum"),
        )
    )
    for k in ("first", "second", "rest"):
        t[f"p_{k}"] = t[f"s_{k}"] / t[f"n_{k}"]
    used = [(0.0, 0.0)] + [
        (round(r.share, 3), round(r.p_second, 3))
        for r in t.itertuples()
        if r.goalies >= MIN_GOALIES
    ]
    return t, used


def holdout(g: pd.DataFrame, tg: pd.DataFrame, starts: pd.DataFrame, season: int, curve) -> None:
    import hockey.serve.schedule as schedule

    saved = schedule.BACK_TO_BACK_SECOND
    schedule.BACK_TO_BACK_SECOND = tuple(curve)
    try:
        rows = []
        for r in g[g.season == season].itertuples():
            sched = tg[(tg.team == r.team) & (tg.season == season)]
            days = [d.date().isoformat() for d in sched["date"]]
            chance = start_chances(r.share, days)
            mine = set(
                starts[(starts.player_id == r.player_id) & (starts.season == season)].game_id
            )
            for gm, d in zip(sched.itertuples(), days, strict=True):
                rows.append(
                    {
                        "player_id": r.player_id,
                        "share": r.share,
                        "kind": gm.kind,
                        "week": gm.date.to_period("W-SUN"),
                        "y": float(gm.game_id in mine),
                        "flat": r.share,
                        "b2b": chance[d],
                    }
                )
    finally:
        schedule.BACK_TO_BACK_SECOND = saved
    d = pd.DataFrame(rows)
    print(f"\nheld out {season}: {d.player_id.nunique()} goalies, {len(d)} goalie-games")
    print("Brier score per game, flat share against back-to-backs counted:")
    for name, sub in [("all nights", d)] + [
        (k, d[d.kind == k]) for k in ("second", "first", "rest")
    ]:
        flat, counted = ((sub.flat - sub.y) ** 2).mean(), ((sub.b2b - sub.y) ** 2).mean()
        print(f"  {name:10s} {flat:.4f}  {counted:.4f}")
    w = d.groupby(["player_id", "week"]).agg(
        y=("y", "sum"),
        flat=("flat", "sum"),
        b2b=("b2b", "sum"),
        share=("share", "first"),
        second=("kind", lambda k: (k == "second").sum()),
    )
    w["role"] = pd.cut(w.share, [0, 0.35, 0.55, 1.01], labels=["backup", "tandem", "starter"])
    print("A fantasy week's starts with a back-to-back in it: actual, flat, counted (mean):")
    sub = w[w.second > 0]
    print(sub.groupby("role", observed=True)[["y", "flat", "b2b"]].mean().round(3).to_string())
    rmse = lambda a: np.sqrt(((a - w.y) ** 2).mean())  # noqa: E731
    print(
        f"root mean squared error, every week: flat {rmse(w.flat):.3f}, counted {rmse(w.b2b):.3f}"
    )


def main() -> None:
    tg = team_games()
    g, starts = goalie_seasons(tg)
    g = g[~g.season.isin(SKIP)]
    table, used = knots(g)
    print(f"{len(g)} goalie-seasons on one team, {sorted(g.season.unique())[0]} on, without {SKIP}")
    print(table[["goalies", "share", "p_first", "p_second", "p_rest"]].round(3).to_string())
    print("knots:", used)
    last = int(g.season.max())
    _, trained = knots(g[g.season < last])
    print("knots without", last, ":", trained)
    holdout(g, tg, starts, last, trained)


if __name__ == "__main__":
    main()
