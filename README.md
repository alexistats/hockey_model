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
python -m hockey.yahoo exchange <code>  # the code from the redirect address bar
python -m hockey.yahoo settings         # league scoring config -> Postgres + YAML
python -m hockey.yahoo crosswalk        # Yahoo player ids -> NHL player ids
python -m hockey.yahoo misses           # review anything it would not guess
```

The authorization code expires in about a minute, so run `exchange` promptly.
After that the refresh token keeps itself alive and none of this repeats.

## Running the model

```bash
python -m hockey.model.mvp --players "Connor McDavid" "Cale Makar"
```

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
