"""Measure what the two-stage shortcut costs.

Stage two holds the shared parameters at stage one's posterior mean, which
treats them as known. That should make intervals slightly too narrow, and a
board whose floors are optimistic is worse than one that is honestly wide - it
is the exact failure the calibration work spent all day removing.

So: fit the same players both ways and compare. The means should barely move.
The interval widths are the number that matters.

    python scripts/validate_staging.py --players 24 --stage-one 40
"""

import argparse
import logging

from hockey.db import SessionLocal
from hockey.features import aging, build_index_maps
from hockey.model import staged
from hockey.model.run import choose_pool
from hockey.model.run_staged import prepare_for
from hockey.yahoo.settings import load_scoring_from_yaml

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=24)
    parser.add_argument(
        "--stage-one",
        type=int,
        default=40,
        help="players used to learn the shared parameters; deliberately a "
        "different, larger set than the ones being compared",
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scoring = load_scoring_from_yaml()

    with SessionLocal() as session:
        maps = build_index_maps(session)
        curve = aging.measure(session)
        pool = choose_pool(session, max(args.players, args.stage_one))
        subject = pool.head(args.players)
        subject_data = prepare_for(session, subject, maps, curve)

        # Shared parameters from a larger, overlapping sample - the same
        # situation a real staged run is in.
        source = pool.head(args.stage_one)
        source_data = prepare_for(session, source, maps, curve)

    logger.info("stage one on %d players", len(source))
    _, params = staged.fit_stage_one(source_data, args.draws, args.tune, args.chains)
    del source_data

    logger.info("joint fit on the %d subject players", len(subject))
    joint_idata, _ = staged.fit_stage_one(subject_data, args.draws, args.tune, args.chains)
    joint_board = staged.projected_board(joint_idata, subject_data, scoring)
    joint_diag = staged.diagnose(joint_idata, "joint")
    del joint_idata

    logger.info("staged fit on the same %d players", len(subject))
    staged_idata = staged.fit_batch(subject_data, params, args.draws, args.tune, args.chains)
    staged_board = staged.projected_board(staged_idata, subject_data, scoring, shared=params)
    staged_diag = staged.diagnose(staged_idata, "staged")
    del staged_idata

    comparison = staged.compare_intervals(joint_board, staged_board)
    print("\n=== convergence ===")
    print(f"  joint  {joint_diag}")
    print(f"  staged {staged_diag}")
    print("\n=== what the shortcut costs ===")
    print(staged.summarise_comparison(comparison))
    print("\n=== per player ===")
    show = comparison[
        ["mean_joint", "mean_staged", "mean_diff", "width_joint", "width_staged", "width_ratio"]
    ]
    print(show.round(2).to_string())

    out = "artifacts/staging_validation.csv"
    comparison.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
