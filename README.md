# hockey_models

An NHL data warehouse feeding an explainable Bayesian projection model, tuned
to one Yahoo fantasy league.

The output is a **variance description** per player - floor, ceiling,
confidence, and head-to-head P(A beats B) - not a single-point projection. It
re-runs to condition on games already played and sharpen rest-of-season value,
and it feeds a separate draft-bot project.

## Setup

```bash
docker compose up -d                    # Postgres 16 on host port 5434
uv venv --python 3.12
uv pip install -e ".[dev,model]"
cp .env.example .env                    # then fill in the Yahoo values
alembic upgrade head
```

## Loading the warehouse

```bash
python -m hockey.ingest backfill        # 2018-19 to 2025-26, a few hours
python -m hockey.ingest schedule --season 20262027
python -m hockey.ingest players --season 20262027
python -m hockey.ingest injuries        # ESPN, feeds the availability model
python -m hockey.ingest.coverage        # what actually landed, and any gaps
```

The backfill is idempotent and resumable. If it stops, run it again and it
picks up from the games that already have logs.

## Connecting Yahoo

You need a Yahoo developer app (Client ID and Secret) and your league id. The
league id is the number in your league URL:
`https://hockey.fantasysports.yahoo.com/hockey/<id>`.

```bash
python -m hockey.yahoo auth-url         # open the URL it prints, approve
python -m hockey.yahoo exchange <code>  # the code from the redirect URL
python -m hockey.yahoo settings         # league scoring config -> Postgres + YAML
python -m hockey.yahoo crosswalk        # Yahoo player ids -> NHL player ids
python -m hockey.yahoo misses           # review anything it would not guess
```

`YAHOO_REDIRECT_URI` must be one of the Redirect URIs registered on the Yahoo
app, matched exactly including the port. A mismatch answers `/oauth2/error`
with `invalid redirect uri` instead of showing a login page.

Do not substitute `oob` for an app that has a real callback registered.
Yahoo accepts it as far as the login page and issues a token that every API
call then refuses with a 403, which looks exactly like a missing permission.

The code expires in about a minute, so run `exchange` promptly. After that the
refresh token keeps itself alive and none of this repeats.

## Running the model

```bash
python -m hockey.model.mvp --players "Connor McDavid" "Cale Makar"

# The full draft board: ~295 skaters, quota-filled per position, ~18 minutes.
python -m hockey.model.run_staged --pool 300 --stage-one 40 --batch 50 \
    --draws 1000 --tune 800 --chains 4 --out artifacts/board

# Replacement level, value over replacement, tiers and the scarcity curve.
python -m hockey.export artifacts/board

# The single-file draft interface.
python scripts/build_draft_ui.py artifacts/board
```

## On a second machine

Three things the clone does not carry, in the order they cost time.

**The warehouse.** The database lives in a Docker volume, not in git. Running
`python -m hockey.ingest backfill` from scratch is several hours of polite API
calls. Copying it across is minutes, and compresses to about 4 MB:

```bash
# on the machine that has it
docker compose exec -T db pg_dump -U hockey hockey_models | gzip > hockey.sql.gz

# on the new one: bring Postgres up, then restore into the empty database
docker compose up -d
gunzip -c hockey.sql.gz | docker compose exec -T db psql -U hockey hockey_models
```

Restore *instead of* `alembic upgrade head`, not after it. The dump carries the
schema and the `alembic_version` row with it, so migrating first leaves tables
for the restore to collide with. Run `alembic upgrade head` afterwards only to
confirm it reports nothing to do.

**`.env`.** Copy it by hand; it is gitignored because it holds the Yahoo client
secret. `.env.example` lists every key it needs.

**A fitted board.** `artifacts/board_*/` holds the posterior draws, roughly
15 MB per run. Copy the directory to skip a refit, or just rerun the board -
it is under 20 minutes.

## Development

```bash
pytest
ruff check . && ruff format --check .
python scripts/generate_data_dictionary.py    # after any schema change
```

## Where to read next

- `CLAUDE.md` - conventions, and the things that are easy to get wrong here
- `docs/architecture.md` - every structural decision and why it was made
- `docs/data-dictionary.md` - generated from the schema's own comments
