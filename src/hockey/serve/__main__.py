"""Run the draft API.

    python -m hockey.serve --board artifacts/board_v3 --goalies artifacts/goalies_v2

Binds to localhost only. Nothing here authenticates anything, because nothing
here is meant to leave this machine.
"""

import argparse
import logging
from pathlib import Path

import uvicorn

from hockey.serve.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.serve")
    parser.add_argument("--board", default="artifacts/board_v3")
    parser.add_argument("--goalies", default=None)
    parser.add_argument("--teams", type=int, default=14)
    parser.add_argument("--slot", type=int, default=8, help="my draft position, 1-based")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    app = create_app(
        Path(args.board),
        None if args.goalies is None else Path(args.goalies),
        n_teams=args.teams,
        slot=args.slot,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
