"""Which seasons the model fits on, and which one it projects.

These are one definition in one place because several parts of the system have
to agree on them exactly. The season being projected must not appear in the
fitting window: it is the extra step the random walk takes past the last
observed season, and if it were also indexed as an observed season the model
would treat "no games played yet" as "played and scored nothing".
"""

# 2018-19 through 2025-26. Eight seasons, chosen so the window reaches back
# past the two the pandemic distorted: 2019-20 stopped at 1082 of 1271 games,
# and 2020-21 was a planned 56-game season on divisional-only schedules. Rates
# per game are unaffected by a short season; games played is not, which is why
# the availability component reads each season's real length rather than
# assuming 82.
FITTING_SEASONS = [
    20182019,
    20192020,
    20202021,
    20212022,
    20222023,
    20232024,
    20242025,
    20252026,
]

# The season being drafted for.
PROJECTION_SEASON = 20262027

# Nominal season length, used only for display and for the coverage report's
# expectations. The availability component does NOT read this: it counts the
# games each player's own team actually played, because in 2019-20 the teams
# differ from each other - between 66 and 74 before the stoppage - so no
# single number is right for that season.
SEASON_LENGTH = {
    20182019: 82,
    20192020: 82,  # scheduled; 68-71 were played before the stoppage
    20202021: 56,
    20212022: 82,
    20222023: 82,
    20232024: 82,
    20242025: 82,
    20252026: 82,
    # 84, not 82. The collective agreement expands the regular season to 84
    # games starting in 2026-27, which the published schedule confirms: 1344
    # regular-season games, 42 home and 42 away for every team. Every
    # counting-stat projection for this season is over 84 games, so treating
    # it as 82 would understate every player by about two and a half percent.
    20262027: 84,
}


def season_label(season: int) -> str:
    """20242025 -> '2024-25'."""
    return f"{season // 10000}-{season % 10000 % 100:02d}"
