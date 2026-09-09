"""Turning a stat line into fantasy points under a league's own scoring config.

Pure functions with no database, no network and no global state. Everything
these read comes from the `ScoringRule` list handed in, which is built from the
`league_stat_categories` table, which is written from Yahoo's API. Nothing here
knows that goals are worth 5 - that is a fact about one league, not about
hockey.

Two entry points, deliberately parallel:

- `score_statline` for one game or one season total, taking a mapping.
- `score_draws` for posterior draws, taking arrays. The model produces tens of
  thousands of draws per player per category, so scoring them one at a time is
  not viable; both go through the same rule list so they cannot disagree.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from hockey.scoring.categories import Category, lookup


class ScoringConfigError(ValueError):
    """The league scores a category this project cannot compute correctly."""


@dataclass(frozen=True)
class ScoringRule:
    """One category and what a unit of it is worth."""

    category: Category
    modifier: float

    @property
    def key(self) -> str:
        return self.category.key


@dataclass(frozen=True)
class LeagueScoring:
    """A league's complete scoring rules, split by position type.

    Skater and goalie rules are kept apart because a stat line only ever
    belongs to one of them, and quietly applying a goalie rule to a skater
    (or the reverse) would produce a number that looks fine and is wrong.
    """

    league_key: str
    skater_rules: tuple[ScoringRule, ...]
    goalie_rules: tuple[ScoringRule, ...]

    @property
    def skater_keys(self) -> tuple[str, ...]:
        return tuple(r.key for r in self.skater_rules)

    @property
    def goalie_keys(self) -> tuple[str, ...]:
        return tuple(r.key for r in self.goalie_rules)


def build_scoring(
    league_key: str,
    categories: Sequence[tuple[str, float]],
) -> LeagueScoring:
    """Build the scoring rules from (display_name, modifier) pairs as Yahoo
    reports them.

    Refuses, rather than guessing, in three cases:

    - an abbreviation the catalogue has never seen (UnknownCategoryError),
    - a category the warehouse cannot serve, which would otherwise score as a
      silent zero for every player,
    - a rate category such as SV% or GAA, which cannot be summed across games
      and so cannot be scored by this module's per-game math.

    A category with a zero modifier is dropped, not checked: Yahoo lists every
    category it knows in `stat_categories` and gives the ones a league does not
    use a modifier of 0. Only the scored ones have to be servable.
    """
    skater: list[ScoringRule] = []
    goalie: list[ScoringRule] = []
    unservable: list[str] = []
    rates: list[str] = []

    for display, modifier in categories:
        if modifier == 0:
            continue
        category = lookup(display)
        if category.is_rate:
            rates.append(f"{display} ({category.key})")
            continue
        if category.source is None:
            unservable.append(f"{display} ({category.key})")
            continue
        rule = ScoringRule(category=category, modifier=float(modifier))
        (goalie if category.position_type == "G" else skater).append(rule)

    problems = []
    if unservable:
        problems.append(
            "the warehouse has no column for: "
            + ", ".join(unservable)
            + ". Add the column and its ingest, or the projection will silently "
            "score these as zero."
        )
    if rates:
        problems.append(
            "these are rate stats and cannot be summed across games: "
            + ", ".join(rates)
            + ". Scoring them per game would produce a plausible wrong number."
        )
    if problems:
        raise ScoringConfigError(f"league {league_key}: " + " Also, ".join(problems))

    return LeagueScoring(
        league_key=league_key,
        skater_rules=tuple(skater),
        goalie_rules=tuple(goalie),
    )


def _rules_for(scoring: LeagueScoring, position_type: str) -> tuple[ScoringRule, ...]:
    if position_type == "G":
        return scoring.goalie_rules
    if position_type == "P":
        return scoring.skater_rules
    raise ValueError(f"position_type must be 'P' or 'G', got {position_type!r}")


def score_statline(
    scoring: LeagueScoring,
    statline: Mapping[str, float | None],
    position_type: str,
) -> float:
    """Fantasy points for one stat line, keyed by canonical category key.

    A missing or NULL stat counts as zero, which is the right reading of the
    warehouse: `ppp` is NULL for a skater who dressed but never left the bench,
    and that skater did score zero power-play points. A category the league
    scores but the stat line omits entirely is also zero, for the same reason -
    a game log with no `shp` key is a game with no shorthanded points.
    """
    total = 0.0
    for rule in _rules_for(scoring, position_type):
        value = statline.get(rule.key)
        if value is not None:
            total += rule.modifier * value
    return total


def score_draws(
    scoring: LeagueScoring,
    draws: Mapping[str, np.ndarray],
    position_type: str,
) -> np.ndarray:
    """Fantasy points for posterior draws.

    `draws` maps a category key to an array of draws. Every scored category
    must be present and every array the same shape, because the whole point is
    to preserve the correlation between categories across draws: a draw where a
    player scored three goals is the same draw where they took eight shots, and
    summing them per draw keeps that relationship. Averaging first and scoring
    the averages would throw it away and shrink the variance.
    """
    rules = _rules_for(scoring, position_type)
    if not rules:
        raise ScoringConfigError(f"league {scoring.league_key} scores no {position_type} category")

    missing = [r.key for r in rules if r.key not in draws]
    if missing:
        raise KeyError(
            f"posterior draws are missing scored categories: {', '.join(sorted(missing))}. "
            f"Scoring without them would understate every player equally and look plausible."
        )

    arrays = [np.asarray(draws[r.key]) for r in rules]
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(
            f"draw arrays have mismatched shapes {sorted(shapes)}; they must be aligned "
            f"draw-for-draw or the correlation between categories is lost."
        )

    total = np.zeros(arrays[0].shape, dtype=float)
    for rule, array in zip(rules, arrays, strict=True):
        total += rule.modifier * array
    return total
