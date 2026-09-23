# Architecture decisions

One entry per structural decision: what, why, and what was considered instead.

---

## The warehouse is lifted from the Fantasy_hockey_app repo

**What.** `models/nhl.py`, the whole of `ingest/`, and the config and session
setup were copied from `Fantasy_hockey_app/fantasy-hockey/backend/`. The
league, roster, matchup, draft, trade, waiver, notification and user tables
were left behind.

**Why.** That ingest layer is already idempotent, resumable, polite to the API
and tested. Rewriting it would have reproduced its bugs without its tests. No
foreign key runs from an NHL table to a league table, so the cut is clean; the
references all point the other way.

**Instead.** A fresh Alembic chain rather than copying the app's version files,
whose `down_revision` graph threads through league migrations.

**Kept.** `player_injuries`, because it feeds the availability component.
**Dropped.** `player_eligibility`, because Yahoo is the authority on
eligibility for this league, not ESPN.

---

## Game logs merge two NHL API sources, and one column was missing

**What.** A skater game log is assembled from the boxscore
(`/gamecenter/{id}/boxscore`) and then completed from the per-player game log
(`/player/{id}/game-log/{season}/{type}`).

**Why.** Neither endpoint is sufficient. Verified directly against the API on
2026-09-09:

| Category | Boxscore | Player game log |
|---|---|---|
| Goals, assists, +/-, SOG | yes | yes |
| **Hits, blocks** | **yes** | no |
| **PPP, SHP** | no | **yes** |

**The change from the app's schema:** `skater_game_logs.shp` is new. The app
never stored shorthanded points because its league did not score them. This one
scores SHP at 1.5, so the per-player pass now fills `ppp` and `shp` together.
Everything else the league scores was already there.

Goalie categories are all served: `started`, `wins`, `goals_against`, `saves`
and `shutouts` cover games started, wins, goals against, saves and shutouts.

---

## `team_abbrev` on the game logs

**What.** Both game-log tables record the team the player dressed for in that
specific game, taken from the side of the boxscore they were listed under.

**Why.** The model needs the opponent and whether the game was at home, and
neither is recoverable without knowing the player's side. `players.team_abbrev`
holds one current value, so it is wrong for every past season and for anyone
traded mid-season. This was the one addition the app's schema needed for
modelling rather than for league play.

---

## Historical franchises drive historical syncs

**What.** `sync_teams(season=...)` reads `/standings/{date}` for a date inside
that season, and the abbreviations it returns drive that season's roster and
schedule calls.

**Why.** The franchise list changes inside the fitting window. Seattle arrived
in 2021-22; Arizona became Utah in 2024-25. A backfill driven off today's team
list 404s on `/roster/SEA/20182019` and never fetches Arizona at all, losing
six seasons of Coyotes games. The standings-by-date endpoint is the only source
that reports the franchises of a past season.

February 1 of the end year is inside every season in the window, including
2020-21, which did not start until January 13, 2021. A season that has not been
played has no standings on any date, so the upcoming season falls back to
today's teams.

Historical teams are upserted and never removed: `nhl_games` rows for 2018-19
Arizona games carry a foreign key to `nhl_teams.abbrev = 'ARI'`.

---

## Eight seasons, 2018-19 to 2025-26

**What.** The fitting window is fixed in `hockey/seasons.py`.

**Why.** Six seasons would start at 2020-21, and half of that window is
pandemic-distorted: 2019-20 stopped at 1082 of 1271 games, and 2020-21 was a
planned 56-game season on divisional-only schedules. Eight reaches back to two
clean seasons so the random walk can see through the distortion.

Verified data floors: player game logs and boxscores go back to 2005-06; full
play-by-play only to 2009-10 (2008-09 carries scoring plays only).

---

## Play-by-play is deferred

**What.** The `nhl_plays` table and its ingest ship, but `backfill` does not
run them. One command enables it.

**Why.** Its purpose here would be deriving faceoff wins, and this league does
not score them. Skipping it avoids roughly two hours of ingest and 3.3 million
rows that nothing would read. Hits and blocks, the other things one might reach
to play-by-play for, come from the boxscore.

---

## The Yahoo config drives the scoring, and the code never does

**What.** Categories and modifiers are read from Yahoo into
`league_stat_categories`, and `hockey/scoring/` builds its rules from that
table. A YAML snapshot is written alongside each sync.

**Why.** Hardcoding "goals are worth 5" states a fact about one league in one
season as if it were a fact about hockey. Settings change, and a change should
appear as a diff rather than as a mystery.

**Three things it refuses to do**, all at config-load time rather than
silently:

- an abbreviation the catalogue has never seen,
- a category the warehouse has no column for, which would otherwise score as
  zero for every player and look entirely normal,
- a rate stat such as SV% or GAA, which cannot be summed across games. Scoring
  one per game produces a plausible wrong number.

---

## Identity is never guessed

**What.** The Yahoo crosswalk folds names, indexes our players by the folded
pair, accepts a single match, breaks a tie on team, and records anything still
ambiguous as a miss with `nhl_id` NULL.

**Why.** A wrong id does not look wrong. It produces a complete, plausible
projection attributed to the wrong player, and nothing downstream can detect
it. A recorded miss is visible and fixable before the draft. This is the method
the app repo proved on ESPN injuries, where it matched 99 of 105.

---

## The model: what was kept and what was fixed

The architecture from the original `game_level_modelling.py` is kept
deliberately - a Gaussian random walk for the player's latent rate across
seasons, an AR(1) opponent process, a home effect, and Poisson counts on a log
link. That structure is what produces a calibrated interval instead of a point
estimate.

Five changes, each fixing a defect rather than changing the approach:

1. **Players are fit jointly.** The original estimated a 31-by-seasons array of
   opponent effects inside a single player's model - roughly 250 parameters
   from about 600 games. It could not identify team strength and mostly
   absorbed noise. Fit across players, the same AR(1) becomes a real
   league-level effect.

2. **Partial pooling by position.** A player's walk starts from a
   position-level mean, so a thin-data player shrinks toward their peer group
   instead of producing an unstable, uselessly wide posterior. The original had
   no pooling.

3. **The assists likelihood no longer conditions on that game's goals.** The
   original regressed assists on goals scored in the same game - a quantity
   that does not exist for a future game. Its own forecast silently dropped the
   term, so the model that was fit was not the model that predicted.

4. **The projected season is a real step in the walk.** The original froze the
   latent state at the last observed season. Here the walk and the AR process
   each take one more step, with no observations, so year-to-year drift
   uncertainty reaches the forecast. At draft time that drift is most of the
   honest uncertainty about a player.

5. **The team count comes from the data.** The original hardcoded 31 against a
   32-row mapping file in a 32-team league.

Two implementation choices that are not changes of approach:

- **Non-centered parameterization** for the random walk. The centered form is
  a funnel that NUTS samples badly. Same model, written so the sampler can
  move.
- **An explicit `init_dist` on the AR process.** PyMC otherwise defaults to
  `Normal(0, 100)`, which on a log rate allows a starting opponent effect of
  e^100.

---

## Inference and forecasting are separate

**What.** The model graph contains no prediction variables. NUTS samples the
continuous parameters; `hockey/model/forecast.py` draws the remaining
schedule's counts from the posterior afterwards.

**Why.** Two reasons.

Projected counts are discrete, and NUTS cannot sample discrete free variables.
The original got them only because PyMC's default sampler quietly assigned them
to Metropolis, which mixes poorly - so the forecast was the worst-sampled part
of the model.

And it is what makes the daily re-run affordable. Conditioning on games played
since the last fit is a forward simulation over a shortened schedule, seconds
of work, not a refit. The plan is a full NUTS fit weekly overnight with daily
re-projection between fits, and any number produced by the fast path will be
labelled as such.

---

## 2026-27 is an 84-game season

Not 82. The collective agreement expands the regular season starting in
2026-27, and the published schedule confirms it: 1344 regular-season games, 42
home and 42 away per team. Treating it as 82 would understate every counting
projection by about two and a half percent.

## Positional value lives outside the model

The model projects stat lines. It knows nothing about rosters, and it should
not: the number that decides a pick is not a player's total but the gap between
that player and whoever would otherwise fill the slot, and that gap changes
with the league's shape rather than with any player's evidence.

So `src/hockey/export/replacement.py` re-reads a finished posterior against the
roster. Replacement level per position, value over replacement, tier breaks and
the positional drop-off curve all come from the saved draws, and every one of
them can be recomputed in milliseconds when the roster assumption changes. The
draft interface does exactly that as players come off the board: a position
that empties makes everyone left at it worth more, and a value column baked in
at fit time cannot say so.

Two decisions inside it are judgement rather than arithmetic, and both are
parameters.

Replacement sits past the *drafted* pool, not past the starting lineup. This
league starts 12 and rosters 17, so 70 bench slots are filled before anyone
reaches the waiver wire. The bench is charged to positions in proportion to
starts, which is how managers actually carry spares.

Tiers break on the probability that the next player outscores the one who
opened the tier, not on the gap between their projections. Two players five
points apart are a coin flip or a real step down depending entirely on how wide
they are, and a gap-based tiering cannot tell those apart.

## The draftable pool is chosen by league points, per position

The pool was first ranked by goals and assists, and that was wrong twice.

It ignored hits and blocks. Both are worth half a point in this league, and
over two seasons a checker's hits outweigh a 20-goal season. Defencemen and
grinders were being excluded on categories the league barely scores.

Worse, it fixed the positional mix by accident. Centres are the deepest scoring
position, so a single ranked list of 300 returned 112 of them and 64
defencemen - for a league that drafts 84 defencemen. Replacement level at
defence then fell outside the pool entirely and had to be read off the pool's
worst players, which overstated every defenceman on the board.

The pool is now ranked on fantasy points actually scored under the league
config, and filled to a multiple of the slots the league drafts at each
position. Stage one's sample is balanced the same way, because it estimates the
per-position baselines and was previously fitted on whichever few defencemen
scored like forwards.

## The room drafts in its own order, and the cost of waiting is simulated

`cost_of_waiting` answers one question at each of my turns: if I pass on a
position now, what is the best player there worth when my turn comes back? The
answer is entirely about what the other managers take in between, and the first
version assumed they draft straight down our value board. They cannot - they
have never seen it.

Measured on the saved Yahoo mock rooms (1,923 picks by other managers):

- From round 4 on the median room pick was the 20th-35th best player left on
  our board, and within a position the room took our best player there only
  6-16% of the time.
- 84-100% of room picks through round 12 filled an open starting slot for the
  team picking. About one pick in seven was a player not on our board at all.
- A player's average pick over the other rooms predicts the next room pick far
  better than our value: held out one room at a time, 3.15 nats per pick on the
  log of the average pick, against 4.28 for our value. Given the average pick,
  our value adds nothing.

So the straight run priced waiting on C, LW, RW and G 25-50 points too high on
every turn, and read defence as free for thirteen rounds of one mock because no
defenceman sat in the top dozen of our board. Replacing it with the brief's
suggestion - each team takes our best player at a position it still needs -
made defence worse, not better: the room does fill its slots, but not with our
best player.

`hockey.serve.room` models the room as it measured. Each pick is a multinomial
logit over the players left: the log of the player's average pick, whether he
fills an open starting slot for the team picking (read off the observed draft
order, and only after that order is checked against my own picks), and a
constant for somebody off our board. The costs are expectations over 500
simulated rooms, scored in our value. Out of sample, on the bot's own turns, the
bias is under 2 points at every position, where the straight run's was -25 to
-51 at four of five.

The order comes from mock rooms, which run Yahoo's standard settings, not this
league's. That is the known weakness. It is still the room's own behaviour
rather than a board none of the other managers can see, and rebuilding it is one
command once more mocks are saved.

The draft page has no server behind it, so it carries the same simulation in
JavaScript. That is a second implementation of the kind this project keeps
having to undo, so the page keeps it between markers and `tests/test_room.py`
runs that block under Node against the Python on the same room: same expected
best-after, same survival odds, to within a few thousand simulated rooms' noise.
When the rosters cannot be read - the page often cannot, because a pick of
somebody off the board cannot be marked - both use coefficients fitted without
the need term rather than switching the term off.
