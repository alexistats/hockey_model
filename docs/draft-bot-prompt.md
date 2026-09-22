I liked that it picked a full team, and respected the goalie draft order. However,
the selections seem odd. It picked all forwards, then 3 goalies, then 4
defensemen. And when I look at the picks, Cozens in the 3rd round looks
suspicious with guys like Dahlin and Schaefer on the board, or a better forward
like DeBrincat.

I know why. I had asked for conservative picks early (floors) and higher-ceiling
picks later, and floors and ceilings are **raw points**. Raw floors put nearly
every defenceman below the forwards, because defencemen score fewer points.
Raw ceilings put goalies above everyone, because goalies are the most volatile
position. So "floors first, ceilings later" on raw numbers produces exactly
forwards, then goalies, then defence.

The idea is still right, but it has to be measured against replacement level,
the same way VORP is. The draft server now does that for you. Please stop
ranking players yourself and follow the procedure below.

## The server

The valuation runs as a local HTTP service in `hockey_models`. Do not recompute
VORP, replacement level, tiers, risk scores or roster assignment. The server
does all of it live against the current draft, and a second copy drifts.

    python -m hockey.serve --board artifacts/board_v3 --goalies artifacts/goalies_v2 --slot 8

Base URL `http://127.0.0.1:8899`. Interactive docs at `/docs`.

| Call | When | What you use |
|---|---|---|
| `GET /health` | once at start | `convergence` (r-hat); refuse to draft if missing |
| `POST /draft/reset` | before a mock | clears state |
| `POST /draft/observed` | every time the board changes | body `{"names": [...all drafted...], "mine": [...my players...]}`; check `unresolved` |
| `GET /recommend` | my turn | everything needed to pick (below) |
| `GET /team/me` | my turn, for the log | `lineup`, `needs`, `bench` |
| `GET /compare?ids=a,b` | optional, close calls | P(A outscores B) |

`/draft/observed` takes the **complete** list of drafted players every time, not
the latest pick. The server works out what changed, so a missed or repeated poll
does no harm. If `unresolved` is non-empty, **stop and ask me**. Never guess a
player's identity, because a wrong match looks completely plausible and nothing
downstream can catch it.

## What `/recommend` returns

- `candidates`: up to 4 players the rules allow, best first, ranked by
  `ranked_by`.
  - With `ranked_by: "risk_score"` (the default), each player's score is
    `(1-w)*(p20 - replacement) + w*(p80 - replacement)`, where `w` is
    `risk_weight`. The weight goes from 0.2 in round 1 (cautious) to 0.8 in the
    last round (upside), so this is my floor-to-ceiling strategy measured
    against replacement.
  - Each candidate also carries `vorp`, `mean`, `p20`, `p80`, `floor`,
    `ceiling`, `tier`, `fills_a_need`, `beats_next` and `notes`.
- `positional_read`: whether a position is emptying out before my next turn.
  - `call` is one of `"position"`, `"no_clear_call"` or `"best_available"`.
  - When `call` is `"position"`, `position` and `take` name the best player the
    rules allow at that position.
  - It also carries `reason`, `open_costs` and `urgent`. Positions a rule blocks
    right now are already excluded.
- `cost_of_waiting`: for each position, the best value now against the best
  expected after the picks before my next turn.
- `blocked_by_rules` and `rule_cost`: who a goalie rule stopped me from taking,
  and what that cost.

The goalie rules are enforced by the server, not by you:
- first goalie from round 7,
- second from round 14,
- at most 3 goalies in total.

## How to choose a pick

1. **Sync.** `POST /draft/observed` with everything visible on the board. Stop
   if anything is unresolved.
2. **Read.** `GET /recommend`.
3. **Decide:**
   - If `positional_read.call == "position"` and `take` is not null, take
     `take`. That position is losing value fast enough to be worth reaching for.
   - Otherwise take `candidates[0]`.

   That is the whole rule. Do not add a positional preference of your own, do
   not fill positions in roster order, and do not re-rank candidates by raw
   floor, ceiling or mean.
4. **Record.** If the draft page has not reflected my pick by the next poll,
   `POST /draft/mine` with the `player_id`.

You may deviate only to ask me, never silently. If something looks wrong, say
so. Examples: a candidate's notes say its value is a lower bound, or a player's
`fills_a_need` is false while another candidate's is true and they are close.

## Things that will mislead you

- **Never add up VORP or risk scores across a roster.** Each player's value
  assumes their best eligible slot, so flexible players are counted twice. Use
  `team/me.lineup` to see a roster.
- `projected_points` in `/team/me` is a plain sum of means, bench included. It is
  not a lineup score.
- `cost_of_waiting` assumes the room drafts straight down the value board. It is
  a direction and rough size, not a forecast.

## Log every pick

One line per pick, so a mock can be audited afterwards:

    R3 #36  took Matthew Schaefer (D)  vorp 204  score 177  w 0.28
            why: candidates[0]  (positional_read: no_clear_call - "passing on D costs about 26 and on C about 21; ...")
            next best: DeBrincat 187, Johnston 181
            rule_cost: none

`why` must be either `candidates[0]` or `positional_read -> <position>`. Anything
else is a bug. Report it rather than keep the pick.

After the next mock, send me the full log.
