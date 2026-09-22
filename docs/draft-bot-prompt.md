# Brief for the draft bot

The valuation lives in `hockey_models` and is served over HTTP. Your job is to
watch the draft page, tell the server what you see, and make the pick it
returns. Do not rank players yourself: do not recompute value over replacement,
replacement level, tiers, risk scores, roster assignment or schedule fit. The
server does all of it live against the current draft, and a second copy of that
logic drifts from this one silently.

    python -m hockey.serve --board artifacts/board_v3 --goalies artifacts/goalies_v2 --slot 8

Base URL `http://127.0.0.1:8899`. Interactive docs at `/docs`.

## The loop

| Call | When | What you use |
|---|---|---|
| `GET /health` | once at start | `convergence` (r-hat). No convergence block, no draft. |
| `POST /draft/reset` | before a mock | clears state |
| `POST /draft/settings` | once, if the draft is not 17 rounds | `{"rules": {"draft_rounds": 16}}` — every deadline is measured against it |
| `POST /draft/observed` | every time the board changes | `{"names": [...everyone drafted...], "mine": [...my players...]}` |
| `GET /recommend` | my turn | `recommendation`, and everything behind it |
| `GET /team/me` | my turn, for the log | `lineup`, `needs`, `bench` |
| `GET /compare?ids=a,b` | close calls worth explaining | P(A outscores B) |
| `POST /draft/mine` | after my pick, if the page has not shown it yet | records it |

`/draft/observed` takes the **complete** list of drafted players every time, not
the latest pick. The server diffs it, so a missed poll, a double poll or a
reconnect all heal themselves. If `unresolved` comes back non-empty, **stop and
ask a human**. Never guess an identity: a wrong id produces a complete,
plausible projection for the wrong player and nothing downstream can detect it.

## Making the pick

**Take `recommendation.player_id`.** That is the whole rule.

It already applies, in this order: a position draining fast enough to reach for,
then an open starting slot, then the job the plan gave this pick, then the best
score. `recommendation.why` names which rule fired, and `detail` says why.

Do not re-rank the candidates. Do not add a positional preference. Do not pick
by raw floor, ceiling, mean or tier.

**Deviate only by asking a human, never silently.** Worth raising: a candidate
whose notes say its value is a lower bound; a `beats_next` under 0.55, which is
close to a coin flip; a recommendation carrying an `injury`.

## What is in the response, and what it means

- **`recommendation`** - `player_id`, `player`, `why`, `detail`. Take it.
- **`candidates`** - six by default, best first, ranked by `ranked_by`. Pass
  `?limit=10` for a wider field; it costs nothing.
  - `risk_score` is the ranking number: `(1-w)*(p20 - replacement) + w*(p80 -
    replacement)`, with `w` = `risk_weight` ramping 0.2 in round 1 to 0.8 in the
    last. Cautious early, ambitious late, both measured against replacement so
    positions stay comparable.
  - Also `vorp`, `mean`, `p20`, `p80`, `floor`, `ceiling`, `tier`,
    `fills_a_need`, `need_slot`, `beats_next`, `notes`,
    `replacement_is_lower_bound`, `replacement_free_below`, `injury`,
    `schedule_opening`, `schedule_season`.
- **`roster_pressure`** - `picks_left`, `slots_open`, `slack`, `open` and the
  live `need_first_margin` (`null` means no slack left). Worth logging on every
  pick: it explains a `needs first` that looks early or a `need overridden` that
  looks generous.
- **`plan`** - `pick_kind` is `starter`, `ceiling`, `schedule` or `bench`, with
  `picks_left`. Starting slots come first; once they are full the last five
  picks are planned, and the final two go to schedule fit.
- **`positional_read`** - `call` is `position`, `no_clear_call` or
  `best_available`. Positions blocked by a rule are already excluded.
- **`cost_of_waiting`** - per position, the best available now against the best
  expected at my next turn.
- **`blocked_by_rules`** and **`rule_cost`** - who a goalie rule stopped, and
  what it cost. Log `rule_cost` whenever it is non-null.
- **`context`** - what draft-day data the board carries (schedule span, weeks,
  injury sync date). If `schedule` is missing, the schedule fields will be null
  and you should say so rather than treat them as zero.

Goalie rules are enforced server-side: first goalie from round 7, second from
round 14, three at most.

## The strategy, so you can explain it at the table

**Starting slots before bench depth, on a deadline.** A candidate who fills an
open slot beats one who does not unless the bench player's score clears a
margin — and that margin rises as the draft runs out of picks.

`roster_pressure` carries the arithmetic: `picks_left`, `slots_open` and `slack`
(the difference). The margin starts at 25 and scales by `picks_left / slack`, so
early a clearly better bench player still wins, while with four picks left for
four open slots `need_first_margin` is `null` and no gap buys bench depth at all.

This exists because value alone cannot see the end of the draft.
`cost_of_waiting` correctly reports 0.0 at defence for most of a draft — there is
always another defenceman — but an unfilled starting slot scores zero for the
season. That cost is zero for twelve rounds and then enormous, a shape no value
curve has, so it is counted in picks rather than priced in points.

When a bench player does win, `why` is `need overridden`, and `detail` gives the
gap and the pressure, so the exception is visible rather than silent.

**The shortlist always carries a filler.** When none of the top candidates fills
an open slot, the best player who does is appended, with a note saying why he is
there. A deadline can always be satisfied.

**Positional reach.** When passing on a position costs at least 8 more than the
next open one, take the best player there instead of the best overall.

**The last five picks have jobs.** Three swing for the ceiling, where a bust
costs a waiver claim. The final two are schedule picks.

**Schedule means starts, not games.** `schedule_opening` counts the nights in
the opening two weeks (29 Sept - 11 Oct 2026; week 1 is short) where the player
would actually be in my lineup, filling each night's roster best-first against
the players I already hold. A fourth centre whose games land on nights my first
three already cover starts nothing and is worth nothing that fortnight.
`schedule_season` asks the same over the whole season, which is the right
question for a third centre or fifth defenceman. Each carries `games`, `starts`,
`blocked` and `start_share`.

**Injuries are flagged, not banned.** `injury` carries the ESPN status and when
it was synced. A player who cannot play is skipped for a schedule pick, because
starts are the entire point of that pick. Everywhere else it is reported and the
human decides: the league has two IR slots, so an injured player can be a fine
pick. The ESPN feed matches on name and a few players never match, so a missing
flag is not proof of fitness.

## Things that will mislead you

- **Never add up `vorp` or `risk_score` across a roster.** Each player's value
  assumes his best eligible slot, so flexible players are double-counted. Read a
  roster from `team/me.lineup`.
- `projected_points` is a plain sum of means, bench included. Not a lineup score.
- `cost_of_waiting` assumes the room drafts straight down the value board. It is
  a direction and a rough size, not a forecast.
- **A small edge at a position whose pool has run dry is not an edge.** Centre
  runs out of freely available players around pick 40 in this league: 42 slots
  against about 51 centre-eligible players. When `replacement_free_below` is
  under 6, `replacement_is_lower_bound` is true and the value is understated by
  an unknown amount.
- **Tiers are struck within a position** on raw points, so a tier-5 centre and a
  tier-2 winger are not on one scale. Report it, never override the order for it.
- **Schedule numbers move as my roster fills.** They are computed against the
  roster at the moment you ask. Re-read `/recommend` before every pick rather
  than planning the last five in advance.

## Log every pick

    R16 #217  took Jake Neighbours (LW)  score 95  w 0.76
              why: plan -> schedule — 5 of 5 games in the opening weeks are starts
              plan: schedule, 2 picks left     season: 68/84 starts
              runners-up: Tolvanen 102, Matheson 99
              rule_cost: none     injury: none

`why` is exactly one of these six values. Anything else is a server bug, not a
judgement call: report it rather than keep the pick.

- `positional_read -> <pos>`
- `needs first -> <pos>`
- `need overridden`
- `plan -> schedule`
- `plan -> ceiling`
- `best score`

Send the full log after each mock.
