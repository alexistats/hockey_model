"""Time the sampler under different chain strategies.

JAX sees one CPU device by default, and PyMC's numpyro backend maps chains
across devices, so four chains run one after another on what is effectively one
core - on a 12-core machine. There are two ways out and they are not obviously
ranked, so this measures rather than assumes.

XLA_FLAGS has to be set before JAX is imported, which is why each mode runs as
its own process:

    python scripts/bench_sampler.py --mode parallel
    python scripts/bench_sampler.py --mode vectorized
    python scripts/bench_sampler.py --mode devices
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["parallel", "vectorized", "devices"], required=True)
parser.add_argument("--players", type=int, default=12)
parser.add_argument("--draws", type=int, default=300)
parser.add_argument("--tune", type=int, default=400)
parser.add_argument("--chains", type=int, default=4)
args = parser.parse_args()

# Must happen before anything imports jax.
if args.mode == "devices":
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={args.chains}"
    ).strip()

import pandas as pd  # noqa: E402

from hockey.db import SessionLocal  # noqa: E402
from hockey.features import (  # noqa: E402
    availability_panel,
    build_index_maps,
    skater_panel,
    team_schedule,
)
from hockey.model import multi  # noqa: E402
from hockey.model.run import choose_pool  # noqa: E402
from hockey.seasons import PROJECTION_SEASON  # noqa: E402


def main() -> None:
    import jax
    import pymc as pm

    with SessionLocal() as session:
        pool = choose_pool(session, args.players)
        ids = [int(p) for p in pool["player_id"]]
        maps = build_index_maps(session)
        panel = skater_panel(session, maps, player_ids=ids)
        avail = availability_panel(session, player_ids=ids)
        schedule = pd.concat(
            [
                team_schedule(session, maps, r.current_team, PROJECTION_SEASON).assign(
                    player_id=r.player_id
                )
                for r in pool.itertuples()
            ],
            ignore_index=True,
        )
        data = multi.prepare(
            panel,
            avail,
            schedule,
            maps,
            {int(r.player_id): r.name for r in pool.itertuples()},
            {int(r.player_id): r.position for r in pool.itertuples()},
        )

    model = multi.build(data)
    chain_method = "vectorized" if args.mode == "vectorized" else "parallel"

    start = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=0.95,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": chain_method},
            random_seed=1,
            progressbar=False,
        )
    elapsed = time.perf_counter() - start

    import arviz as az

    diag = az.summary(idata, var_names=["b_home", "loading"], round_to=4)
    print(
        f"RESULT mode={args.mode:11s} devices={jax.local_device_count()} "
        f"chain_method={chain_method:10s} seconds={elapsed:7.1f} "
        f"worst_rhat={diag['r_hat'].max():.4f} "
        f"divergences={int(idata.sample_stats['diverging'].sum())}"
    )


if __name__ == "__main__":
    sys.exit(main())
