"""The parameters every player shares, lifted out of a fitted posterior.

Fitting 400 players jointly costs 19 GB and buys almost nothing over fitting
120, because only about 3,800 values are actually shared: position-level means,
the category loadings, the opponent process, the home effect, the aging scale
and the availability population parameters. Everything else - roughly 161
values per player - is conditionally independent once those are known.
Draisaitl's walk and Suzuki's walk never touch.

So the fit splits. Stage one estimates the shared values from a sample large
enough to pin them down. Stage two holds them and fits players in batches,
which is embarrassingly parallel and whose memory does not grow with the size
of the pool.

The cost is real and worth stating: stage two conditions on stage one's answer
rather than re-estimating it, so it understates uncertainty in quantities that
120 players have already measured well. `validate.py` checks how much by
fitting the same players both ways and comparing the intervals.
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Fixed in stage two. `opponent` is included rather than its rho/sigma/z
# ingredients: holding the process itself drops 1,848 latents and the AR
# machinery with them, and its effect is small enough (sigma_team is about
# 0.04 on a log rate) that freezing its uncertainty costs little.
SHARED_NAMES = (
    "mu_position",
    "sigma_player",
    "sigma_form",
    "loading",
    "sigma_idio",
    "age_scale",
    "opponent",
    "b_home",
    "mu_avail",
    "sigma_avail",
    "phi_avail",
    "sigma_avail_season",
    "kappa_avail",
    "mu_pm",
    "sigma_pm_player",
    "b_home_pm",
    "sigma_pm",
)


@dataclass(frozen=True)
class SharedParameters:
    """Posterior means of the shared values, plus the coordinates they were
    fitted on. The coordinates matter: an opponent process fitted over one set
    of seasons and teams cannot be applied to another."""

    values: dict[str, np.ndarray]
    seasons: list[int]
    teams: list[str]
    stats: tuple[str, ...]
    n_source_players: int
    # Which shared terms the fitting model actually had. Reusing a file fitted
    # with an opponent term in a model without one (or the reverse) would
    # silently apply values estimated under different assumptions, and nothing
    # downstream would look wrong.
    structure: tuple[str, ...] = ()

    def check_compatible(self, maps, stats: tuple[str, ...], structure=None) -> None:
        if structure is not None and tuple(self.structure) != tuple(structure):
            raise ValueError(
                f"these shared parameters were fitted with terms {self.structure}, "
                f"but this model has {tuple(structure)}. Delete shared.npz and refit "
                f"stage one; values estimated under a different model do not carry "
                f"over."
            )
        if list(self.stats) != list(stats):
            raise ValueError(
                f"shared parameters were fitted on categories {self.stats}, "
                f"but this batch uses {stats}"
            )
        if self.seasons != list(maps.seasons):
            raise ValueError(
                f"shared parameters were fitted on seasons {self.seasons}, "
                f"but this batch uses {list(maps.seasons)}"
            )
        if self.teams != list(maps.teams):
            raise ValueError(
                "shared parameters were fitted on a different team list; the opponent "
                "process is indexed positionally and would be misapplied"
            )


def extract(
    idata, maps, stats: tuple[str, ...], n_source_players: int, structure=()
) -> SharedParameters:
    """Posterior means of every shared value in a stage-one fit."""
    posterior: xr.Dataset = idata.posterior
    values = {}
    for name in SHARED_NAMES:
        if name not in posterior:
            if name in ("opponent",) and name not in structure:
                continue  # the model did not have this term
            raise KeyError(
                f"{name!r} is not in the stage-one posterior; it was fitted with a "
                f"different model version than this one expects"
            )
        values[name] = posterior[name].mean(dim=("chain", "draw")).to_numpy()
    logger.info(
        "shared parameters from %d players: %s",
        n_source_players,
        ", ".join(f"{k}{tuple(v.shape) if v.ndim else ''}" for k, v in values.items()),
    )
    return SharedParameters(
        values=values,
        seasons=list(maps.seasons),
        teams=list(maps.teams),
        stats=tuple(stats),
        n_source_players=n_source_players,
        structure=tuple(structure),
    )


def save(shared: SharedParameters, path) -> None:
    np.savez(
        path,
        seasons=np.array(shared.seasons),
        teams=np.array(shared.teams),
        stats=np.array(shared.stats),
        n_source_players=shared.n_source_players,
        structure=np.array(shared.structure),
        **{f"v_{k}": v for k, v in shared.values.items()},
    )


def load(path) -> SharedParameters:
    with np.load(path, allow_pickle=False) as f:
        values = {k[2:]: f[k] for k in f.files if k.startswith("v_")}
        return SharedParameters(
            values=values,
            seasons=[int(s) for s in f["seasons"]],
            teams=[str(t) for t in f["teams"]],
            stats=tuple(str(s) for s in f["stats"]),
            n_source_players=int(f["n_source_players"]),
            structure=tuple(str(x) for x in f["structure"]) if "structure" in f else (),
        )
