# hockey_models

## What this is

A standalone NHL data warehouse feeding an explainable Bayesian projection
model, tuned to one Yahoo fantasy league. The output is a **variance
description** per player - floor, ceiling, confidence, and head-to-head
P(A > B) - not a single-point projection. It re-runs daily to condition on
games already played and sharpen rest-of-season value.

It feeds a separate draft-bot project. It is not that project.

- Design decisions and their reasoning: `docs/architecture.md`
- Schema meanings: `docs/data-dictionary.md` (generated, never hand-edited)
- Model behaviour and calibration evidence: `docs/model-card.md`

## Stack

- Python 3.12, `uv` for the environment, SQLAlchemy 2.x, Alembic, Postgres 16
- PyMC with the **numpyro/JAX** NUTS backend. This machine has no C++
  toolchain, so PyTensor's default backend silently falls back to a
  pure-Python path far too slow to be usable. Do not switch backends without
  checking that first.
- Postgres runs in Docker on **host port 5434**. 5432 is often a native
  install and 5433 belongs to the Fantasy_hockey_app stack; this project must
  never collide with either.

## Layout

- `src/hockey/ingest/` - the NHL and ESPN API clients and sync jobs. **The only
  code allowed to call api-web.nhle.com.** Nothing else may.
- `src/hockey/yahoo/` - the only code allowed to call the Yahoo Fantasy API.
- `src/hockey/models/` - SQLAlchemy models
- `src/hockey/scoring/` - stat line to fantasy points. Pure functions, no I/O.
- `src/hockey/features/` - warehouse to modelling panel
- `src/hockey/model/` - the PyMC model, and the forecast that reads its posterior
- `src/hockey/calibration/` - posterior predictive checks, CRPS, backtests
- `src/hockey/serve/` - the draft-day HTTP API the bot drives. Holds the live
  draft state and re-reads the posterior against it. **Nothing here may change
  a posterior either**; it is the export layer with a socket on it. `room.py`
  models the other managers, fitted on draft_bot's saved mocks into
  `config/room_2026.json`; `cost_of_waiting` is simulated from it.
- `src/hockey/export/` - the posterior read as a draft board. `draft.py` scores
  draws into fantasy points; `replacement.py` re-reads the same posterior
  against the league's roster shape. **Nothing here may change a posterior.**
  Positional value depends on the roster, which changes mid-draft, and a refit
  is never the right answer to that.
- `tests/` - pytest

## Commands

```
docker compose up -d                          # Postgres on 5434
uv pip install -e ".[dev,model]"
alembic upgrade head

python -m hockey.ingest backfill              # 8 seasons, several hours
python -m hockey.ingest.coverage              # what actually landed
python -m hockey.yahoo auth-url               # one-time Yahoo authorization
python -m hockey.yahoo settings               # league scoring config
python -m hockey.yahoo crosswalk              # Yahoo ids -> NHL ids
python -m hockey.yahoo eligibility <paste>    # positions, from a web-UI paste
python -m hockey.yahoo doctor                 # why Yahoo is answering 403
python -m hockey.model.mvp                    # the end-to-end gate
python -m hockey.model.run_staged --pool 300  # the draft board, ~20 min
python -m hockey.model.run_goalies            # the goalie board, ~15 min
python -m hockey.export artifacts/board_v2    # replacement level, value, tiers
python -m hockey.serve.room --mocks ../draft_bot/artifacts/mocks   # the room model
python scripts/trace_cost_of_waiting.py <mock dir> --holdout      # check it on a mock
python scripts/build_draft_ui.py artifacts/board_v2   # the single-file draft page
python -m hockey.serve --board artifacts/board_v3 \n    --goalies artifacts/goalies_v2 --slot 8           # the draft-day API, port 8899

pytest
ruff check . && ruff format --check .
alembic revision --autogenerate -m "msg"
```

## Rules

- **Evidence before claims.** Show model output, coverage numbers and
  calibration tables. Never assert that something works.
- **MVP-first.** Get the narrow version running end to end before widening it.
- **Never guess an identity.** A wrong player id produces a complete, plausible
  projection for the wrong player and nothing downstream can detect it. Record
  the miss and move on. This applies to the Yahoo crosswalk, the ESPN injury
  match, and anything else that joins on a name.
- **Never silently zero a category.** A scoring category the warehouse cannot
  serve is an error at config-load time, not a quiet zero for every player.
- All scoring math lives in `scoring/` as pure, unit-tested functions, driven
  by the league config in the database - never by a constant in the code.
- Database changes go through Alembic. Every migration carries `COMMENT ON`
  for new tables and columns: meaning, units, nullability semantics and data
  source. Regenerate `docs/data-dictionary.md` afterwards.
- Do not switch away from the Bayesian model. A non-Bayesian model may be added
  as a **calibration benchmark only**, never as the primary - point estimates
  cannot produce a calibrated floor and ceiling, which is the whole point.
- Keep the sampler's convergence diagnostics next to any number derived from
  it. A summary from chains that did not converge is not a result.

## Things that are easy to get wrong here

- **The team list is season-specific.** Seattle did not exist before 2021-22,
  and Arizona became Utah in 2024-25. Driving a historical sync off today's
  teams 404s on some and silently omits others. `sync_teams(season=...)` reads
  the standings for a date inside that season.
- **A player's team comes from the game log, not the players table.**
  `players.team_abbrev` is one current value and is wrong for every past
  season and every trade.
- **The projected season is not in the fitting window.** It is the extra step
  the random walk takes past the last observed season. Indexing it as an
  observed season would read "not played yet" as "played and scored nothing".
- **2026-27 is an 84-game season**, not 82, under the new collective agreement.
- **`count` is overloaded in Yahoo's JSON.** It marks collection size and it is
  also the number of a roster slot.
- **The room does not draft off our board.** It drafts in Yahoo's order toward
  its open slots. Anything that predicts other managers' picks from our value
  gets forwards and goalies badly wrong; use the room model.
- **A mock's "best D fell" is not the room's doing if I took him.** Check a
  prediction of the room only at positions my own pick could not fill.
