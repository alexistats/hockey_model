# Brief for the draft bot

Paste this into the draft-bot project as its starting context. It describes
what `hockey_models` hands over, and — more importantly — every way that output
can mislead a consumer that treats it as ground truth.

Written 2026-09-20 against `artifacts/board_v3` (skaters, fitted 2026-09-11)
and `artifacts/goalies_v2` (goalies, fitted 2026-09-20).

---

## What you are consuming

A standalone NHL data warehouse feeding an explainable Bayesian projection
model, tuned to one Yahoo league (id 2270, 14 teams, head-to-head points). The
output is a **variance description** per player — floor, ceiling, confidence,
head-to-head P(A > B) — not a single-point projection.

Roster: C,C,LW,LW,RW,RW,D,D,D,D,G,G + 5 bench + 2 IR.
Scoring: G 5, A 3, +/- 0.5, PPP 0.5, SHP 1.5, SOG 0.5, HIT 0.5, BLK 0.5;
goalies GS 1, W 6, GA −1.5, SV 0.3, SHO 4.

### Files, per board directory

| File | What it is |
|---|---|
| `value_board.csv` | One row per player: `player`, `player_id`, `games`, `mean`, `floor` (p10), `p20`, `p80`, `ceiling` (p90), `spread`, `sd`, `exp_games`, `age`, `position` (NHL primary), `team`, `eligible` (Yahoo), `slot` (valued at), `replacement`, `vorp`, `pos_rank`, `tier` |
| `category_projections.csv` | Per-category mean/floor/ceiling: goals, assists, ppp, shp, sog, hits, blocks, plus_minus |
| `head_to_head.csv` | P(row beats column) over a full season, every pair |
| `replacement_levels.csv` | Per position: slots filled, pool size, replacement level, starter cutoff |
| `scarcity.csv` | Drop-off curve: projected points by rank within position |
| `draws_batch_*.npz` | The posterior itself — `draws` (fantasy totals), `games_played`, `stat_<category>`, all `(n_draws, n_players)` |
| `diagnostics.csv` | Per-batch r-hat, ESS, divergences |
| `config/eligibility_2026.csv` | Yahoo position eligibility, `nhl_id → positions` |

Goalies come from their own board directory (`artifacts/goalies_v2`):

| File | What it is |
|---|---|
| `goalie_board.csv` | Per goalie: mean, floor, p20, p80, ceiling, sd, `exp_starts`, `exp_wins`, `exp_saves`, `exp_ga`, `exp_shutouts`, `save_pct`, `last_season`, `override` |
| `goalie_draws.npz` | Fantasy-point draws only — no per-category arrays |
| `posterior.nc` | The full posterior, saved before anything is computed from it |
| `parameters.csv`, `diagnostics.csv` | r-hat, ESS, divergences |

**Prefer the draws over the summaries.** Every interesting question — P(A > B),
"how often does he clear 300 shots", joint outcomes across categories — is a
pass over `draws_batch_*.npz`. The CSVs are summaries of those draws and throw
away the shape.

---

## Caveats, in rough order of how much damage they can do

### 1. VORP is a per-player maximum and is **not additive across a roster**

`vorp = mean − replacement[best eligible position]`, where each player
independently takes whichever eligible position has the lowest replacement
level. That is correct for *"what is this single pick worth"*, which is what a
draft board answers.

It is **wrong to sum VORP across a drafted roster and call it a team total.**
Two LW/RW-eligible players cannot both occupy the scarcer slot, so flexibility
gets double-counted. If you score whole rosters, you must do your own
capacity-constrained assignment — `hockey.export.replacement.fill_slots` does
exactly this for the baselines and is the function to copy.

### 2. Replacement levels are a snapshot of a full pool, not of the live draft

`replacement_levels.csv` is computed against the **undrafted** pool. The instant
picks start, every number in it is stale. The draft UI recomputes live in
JavaScript; the CSV does not.

If the bot uses static VORP mid-draft it will systematically overvalue whatever
position has already been picked out.

### 3. Tiers mean two different things in two places

- `export/replacement.py` strikes tiers **once over the full pool** (static).
- The draft page re-strikes them **over remaining players** as picks happen, so
  tier 1 refills — when the tier-1 right wings go, the next three are promoted
  into tier 1.

Same word, two behaviours. Decide which you want and do not mix them.

Also: **tier numbers are not comparable across positions.** Tiers are struck
within a position, so a tier-4 defenceman and a tier-4 centre are not
equivalent players. For cross-position comparison use `vorp` directly, which is
already per-position.

### 4. The "cliff" is a raw points gap, not a distributional statement

`scarcity.csv` and the position cards mark the largest single drop between
consecutive ranks by **mean**. Unlike tiers, it ignores the distributions
entirely and says nothing about whether the gap is reliable. Treat it as an
eye-guide, not a signal.

### 5. Age is overstated by one year for ~89% of players

`features/aging.py::season_age` claims to give age at 1 February but does a
plain year subtraction that ignores birth month. Measured: **2,061 of 2,308
players are one year too old**; only the 247 born in January are right.

The same function both *measures* the aging curve and *applies* it, so the
error largely cancels and the projections are not materially wrong. What is
wrong is the labelling — the curve's "past 34" refers to true age ~33 — and the
`age` column in `value_board.csv`. **Do not surface that age to a user, and do
not build age-band logic on it.** Unfixed deliberately: correcting it needs a
re-measure and a refit, for a relabelling rather than better projections.

### 6. Availability barely discriminates

`exp_games` across 295 players: mean 72.8, **sd 4.1**, range 59.2–80.3. The
model hardly distinguishes injury-prone players from durable ones. Any
"availability score" built on it is close to decorative.

The ESPN injury ingest exists (`src/hockey/ingest/`) but **does not feed the
projection**. Current injuries are not priced in at all. If the bot needs
injury awareness it has to come from somewhere else.

### 7. Eligibility is a point-in-time transcription

`config/eligibility_2026.csv` was parsed from a Yahoo web-UI paste on
**2026-09-20**. It is not live.

- Covers **284 of 295** board players; the other 11 fall back to NHL primary
  position, which for some of them is the wrong pool.
- Yahoo eligibility **changes in-season** as players accumulate games at a new
  position. This file will drift and nothing will tell you.
- Regenerate with `python -m hockey.yahoo eligibility <paste file>`.

The flexibility premium is small: across 94 multi-eligible board players the
second position is worth a mean of **4.2 points** (range 3.3–7.3), because the
forward baselines sit within 7.4 points of each other (C 281.4, LW 284.7, RW
288.8). Do not model dual eligibility as a large edge. **D at 209.0 is the gap
that matters** — 72.4 points below C.

### 8. There is no live Yahoo connection, and there will not be one soon

Yahoo closed self-serve Fantasy API access on **22 July 2026** and moved it
behind a manual approval programme at `sports.yahoo.com/developer`. Every
Fantasy endpoint returns 403 "This application is not authorized to perform
this action", for existing apps as well as new ones.

OAuth itself works — identity returns a full profile — so this is policy, not
configuration. `python -m hockey.yahoo doctor` confirms it.

Consequences for the bot:
- **No live draft feed.** Picks must be entered by hand or scraped.
- **No roster sync, no league state, no transaction history.**
- Scoring comes from `config/league_2270.yaml`, transcribed by hand and **not
  verified against live Yahoo**. It runs through the same `build_scoring()` as
  the live sync would, so the model cannot tell the difference — but a
  mid-season settings change would be invisible.

### 9. Goalies exist now, from a **separate model** — read the differences

The pool is 295 skaters + 66 goalies. `python -m hockey.export <board>
--goalies <goalie board>` merges them; without that flag the board is skaters
only and the goalie slots go unpriced.

Merging is legitimate because the two fits are **independent** — goalies share
no parameters with skaters, and independent draws are exactly what P(A > B) is
defined over. Replacement level is a property of the roster, not of how a
player was fitted, so the same functions price both.

What differs, and will bite a consumer that assumes uniformity:

- **Different categories.** Goalies have `exp_starts`, `exp_wins`, `exp_saves`,
  `exp_ga`, `exp_shutouts`, `save_pct`; the skater category columns are NaN for
  them, and vice versa. A NaN here is an absence, not a zero — do not fill it.
- **No per-category draws.** Only goalie *point totals* were saved, so
  `P(A > B)` works on totals but **not** per category. Skaters have both.
- **No age.** The aging curve was measured on skaters and never fitted for
  goalies, so `age` is NaN. Do not infer one.
- **Starts, not games.** `exp_games` for a goalie is expected *starts*.
- **Only goalies who played last season are projected.** The workload walk will
  happily carry a retired goalie to the projected season with a confident
  number attached — the first run of this put Luongo, Lundqvist and Crawford on
  the board, and Ben Bishop at 404 points six seasons after his last game. The
  board now requires an appearance in the most recent season. If you rebuild it
  yourself, keep that filter.
- **Goalie tiers run wide.** Five goalies in tier 1 against one centre, because
  goalie outcomes are uncertain enough that the 40% threshold holds longer. That
  is real information — the top goalies are genuinely interchangeable — not a
  bug to correct.
- **Volume is a refund.** Saves pay 0.3 and goals against cost 1.5, so a shot
  faced breaks even at .833 save percentage and every NHL goalie clears it.
  Facing more shots is always net positive in this league. Do not penalise a
  goalie for a leaky defence without also crediting the volume.
- **`config/goalie_priors.yaml`** can override workload per goalie and nudge
  team shot volume or win rate. Overrides apply as distributions, not point
  estimates. If the bot regenerates the board, it inherits whatever is in there.

Where the variance actually is, measured on 447 goalie-seasons: **71% of a
goalie's season fantasy total is starts**, 17% is per-start quality. Win rate is
the team term that matters (team sd 0.239 on the logit scale, three times the
save-rate terms, on the category worth 6 points). Save percentage splits about
evenly between goalie and team — the commonly quoted "team explains two thirds"
is an artifact of grouping 447 observations into 264 team-seasons; shuffled team
labels "explain" 56.8% of it.

Also: 73 of the 295 skaters are already below replacement, and the board does
not extend past the pool, so deep-bench and waiver players are unpriced —
absence from the board is not evidence a player is bad.

### 10. Sampling caveats

- Skaters, per batch: r-hat **1.006–1.010**, lowest ESS 1,369, **zero
  divergences** on `board_v3`.
- Goalies, `goalies_v2`: **zero divergences**, lowest ESS 240, every global
  parameter r-hat ≤ **1.0096**. One variance component, `sigma_goalie_shots`,
  sits at **1.0115** — it is weakly identified because goalie style barely
  moves shots faced against the team effect, and it has little influence on the
  projection. Stated rather than hidden.
- Keep diagnostics attached to any number you derive — a summary from chains
  that did not converge is not a result.
- Players are fitted in **batches of 50**, conditionally independent given
  shared parameters. Comparing a column from one batch with a column from
  another is valid under that assumption, but they are not the same posterior
  sample, so do not read cross-batch draw pairing as a joint draw.
- Within a batch, draw *k* of a player's goals and draw *k* of their shots
  **are** one season. Pairing is real there and is what makes multi-category
  comparison honest.

### 11. Two small inconsistencies to be aware of

- **Tie handling.** The page splits ties when computing P(A > B); Python's
  `tiers()` uses strict `>`. Category totals are integers and tie often —
  overwhelmingly for shorthanded points, where most seasons are 0 or 1 — so
  counting a tie as a loss biases those comparisons downward. Prefer
  tie-splitting.
- **"Cost of waiting"** in the UI assumes the next N picks come off the top of
  the value board. The room will not do that. It is a direction and rough
  magnitude, not a forecast.

### 12. Season and identity gotchas inherited from the warehouse

- **2026-27 is an 84-game season**, not 82.
- The projected season is *not* in the fitting window — it is the extra step the
  random walk takes past the last observed season.
- A player's team comes from the game log, not `players.team_abbrev`, which is
  one current value and wrong for every past season and every trade.
- **Never guess an identity.** A wrong player id yields a complete, plausible
  projection for the wrong player and nothing downstream can detect it. Record
  the miss. This has already bitten once: Vancouver had two Elias Petterssons,
  a centre and a defenceman.

---

## What the bot should probably do

1. Read the draws, not the summaries, for anything probabilistic.
2. Recompute replacement level and VORP **after every pick**, against the
   remaining pool and the roster shape — never refit. `fill_slots` is the
   capacity-aware primitive.
3. Do its own assignment when scoring a whole roster, per caveat 1.
4. Read goalie rows as a different shape, per caveat 9 - different
   categories, no per-category draws, no age, starts rather than games.
5. Keep r-hat next to anything it reports.

## Regenerating upstream

```
docker compose up -d                               # Postgres on 5434
python -m hockey.model.run_staged --pool 300       # the skater board, ~12 min
python -m hockey.model.run_goalies --tune 2500     # the goalie board, ~25 min
python -m hockey.yahoo eligibility <paste file>    # refresh eligibility
python -m hockey.export artifacts/board_v3 --goalies artifacts/goalies_v2
python scripts/build_draft_ui.py artifacts/board_v3 --goalies artifacts/goalies_v2
```
