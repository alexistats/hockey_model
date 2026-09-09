"""The catalogue of Yahoo scoring categories and where each one is served from.

Yahoo's numeric stat ids are not documented and have shifted between sports and
seasons, so this maps on the abbreviation Yahoo returns alongside every category
(`display_name`) instead. The numeric id is still recorded on the row, but
nothing keys off it.

Every category a league scores must resolve to a `Category` here. One that does
not is an error at config-load time, never a silent zero - see
`hockey.scoring.points.build_scoring`.
"""

from dataclasses import dataclass

# Canonical column names produced by hockey/features/panel.py. The scoring
# functions read statlines keyed by `Category.key`, so these names are the
# contract between the feature layer, the model and the scoring layer.
SKATER = "skater_game_logs"
GOALIE = "goalie_game_logs"


@dataclass(frozen=True)
class Category:
    """One scoring category.

    `source` is the warehouse column that serves it, as 'table.column'. None
    means the warehouse cannot serve it today - that is a hard error if the
    league actually scores the category, and harmless otherwise.

    `is_rate` marks a ratio (save percentage, goals-against average, shooting
    percentage). A rate cannot be scored additively across games: summing a
    per-game save percentage is meaningless. A league that puts a non-zero
    modifier on one of these needs different math than this module implements,
    so `build_scoring` refuses rather than producing a plausible wrong number.
    """

    key: str
    display: str
    position_type: str  # 'P' = skater, 'G' = goalie
    source: str | None
    is_rate: bool = False
    # A category the warehouse computes rather than stores, e.g. points = goals
    # + assists. Recorded so the data dictionary and the coverage report can
    # tell stored columns from derived ones.
    derived_from: tuple[str, ...] = ()


_CATEGORIES: tuple[Category, ...] = (
    # --- skaters ---
    Category("goals", "G", "P", f"{SKATER}.goals"),
    Category("assists", "A", "P", f"{SKATER}.assists"),
    Category("points", "P", "P", f"{SKATER}.points", derived_from=("goals", "assists")),
    Category("plus_minus", "+/-", "P", f"{SKATER}.plus_minus"),
    Category("pim", "PIM", "P", f"{SKATER}.pim"),
    Category("ppg", "PPG", "P", None),
    Category("ppa", "PPA", "P", None),
    Category("ppp", "PPP", "P", f"{SKATER}.ppp"),
    Category("shg", "SHG", "P", None),
    Category("sha", "SHA", "P", None),
    Category("shp", "SHP", "P", f"{SKATER}.shp"),
    Category("gwg", "GWG", "P", None),
    Category("sog", "SOG", "P", f"{SKATER}.sog"),
    Category("shooting_pct", "SH%", "P", None, is_rate=True),
    # Faceoffs are derivable from nhl_plays (details.winningPlayerId /
    # losingPlayerId), but play-by-play is not part of the default backfill
    # because this league does not score them. Enabling it is one command:
    # python -m hockey.ingest play-by-play --seasons ...
    Category("faceoffs_won", "FW", "P", None),
    Category("faceoffs_lost", "FL", "P", None),
    Category("hits", "HIT", "P", f"{SKATER}.hits"),
    Category("blocks", "BLK", "P", f"{SKATER}.blocks"),
    Category("toi", "TOI/G", "P", f"{SKATER}.toi_seconds", is_rate=True),
    # --- goalies ---
    Category("games_started", "GS", "G", f"{GOALIE}.started"),
    Category("wins", "W", "G", f"{GOALIE}.wins"),
    Category("losses", "L", "G", None),
    Category("otl", "OTL", "G", None),
    Category("goals_against", "GA", "G", f"{GOALIE}.goals_against"),
    Category("gaa", "GAA", "G", None, is_rate=True),
    Category("shots_against", "SA", "G", f"{GOALIE}.shots_against"),
    Category("saves", "SV", "G", f"{GOALIE}.saves"),
    Category("save_pct", "SV%", "G", f"{GOALIE}.save_pctg", is_rate=True),
    Category("shutouts", "SHO", "G", f"{GOALIE}.shutouts"),
)

# Yahoo has spelled some of these more than one way across seasons and sports.
# Anything not listed here is looked up by its display name as-is.
_DISPLAY_ALIASES = {
    "+/-": "+/-",
    "PLUSMINUS": "+/-",
    "S": "SOG",
    "SH": "SOG",
    "SHOTS": "SOG",
    "HITS": "HIT",
    "BLKS": "BLK",
    "BS": "BLK",
    "SO": "SHO",
    "SHUTOUTS": "SHO",
    "GST": "GS",
    "TOI": "TOI/G",
    "S%": "SH%",
}

BY_DISPLAY: dict[str, Category] = {c.display: c for c in _CATEGORIES}
BY_KEY: dict[str, Category] = {c.key: c for c in _CATEGORIES}


class UnknownCategoryError(KeyError):
    """Yahoo returned a scoring category this catalogue has never seen."""


def normalize_display(display: str) -> str:
    """Fold a Yahoo abbreviation to the spelling this catalogue uses."""
    cleaned = display.strip().upper()
    return _DISPLAY_ALIASES.get(cleaned, cleaned)


def lookup(display: str) -> Category:
    """The category for a Yahoo abbreviation.

    Raises UnknownCategoryError rather than returning None: an unrecognized
    category the league actually scores would otherwise vanish from every
    projection without a trace.
    """
    key = normalize_display(display)
    try:
        return BY_DISPLAY[key]
    except KeyError as exc:
        raise UnknownCategoryError(
            f"Yahoo category {display!r} (normalized {key!r}) is not in the catalogue. "
            f"Add it to hockey/scoring/categories.py with the warehouse column that "
            f"serves it, or None if nothing does."
        ) from exc
