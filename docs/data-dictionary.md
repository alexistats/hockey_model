# Data dictionary

**Generated** from the schema's `COMMENT ON` text by
`scripts/generate_data_dictionary.py`. Do not edit by hand - change the
comment in the migration and regenerate, or this drifts out of step with
the database and becomes worse than no document at all.

Generated 2026-09-09 against 11 tables.

## `nhl_teams`

NHL teams. Source: GET /v1/standings/now (api-web.nhle.com); numeric id backfilled from /v1/club-schedule-season.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `abbrev` | VARCHAR(3) | no | PK | 3-letter NHL team abbreviation (e.g. TOR). PK; changes on franchise relocation (e.g. ARI->UTA), which would need a data migration. |
| `name` | VARCHAR(64) | no |  | Full team name, English locale (e.g. 'Toronto Maple Leafs'). |
| `conference` | VARCHAR(16) | no |  | Conference name: 'Eastern' or 'Western'. |
| `division` | VARCHAR(16) | no |  | Division name: Atlantic, Metropolitan, Central, or Pacific. |
| `nhl_team_id` | INTEGER | yes |  | NHL numeric team id (e.g. TOR=10). NULL until schedule sync backfills it; the standings endpoint does not provide it. |

## `players`

NHL players. Source: GET /v1/roster/{team}/{season}; minimal stub rows are created from boxscores for players absent from season rosters (call-ups, mid-season trades).

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `nhl_id` | BIGINT | no | PK | NHL player id (e.g. McDavid=8478402). PK. |
| `first_name` | VARCHAR(64) | no |  | First name. For boxscore stub rows this is an abbreviated initial (e.g. 'C.') until a roster sync fills the full name. |
| `last_name` | VARCHAR(64) | no |  | Last name, English locale. |
| `position` | VARCHAR(2) | no |  | Position: C, LW, RW, D, or G. NHL API codes L/R are mapped to LW/RW at ingest (hockey/ingest/mappers.py). |
| `team_abbrev` | VARCHAR(3) | yes | FK nhl_teams.abbrev | FK nhl_teams.abbrev. Team whose season roster listed the player; NULL for boxscore stub rows. |
| `status` | VARCHAR(16) | no |  | 'active' = on a synced season roster; 'inactive' = seen only in boxscores. NOT injury status; that comes from ESPN into player_injuries. |
| `shoots_catches` | VARCHAR(1) | yes |  | Shoots (skaters) or catches (goalies): L or R. NULL for stub rows. |

## `nhl_games`

NHL games, regular season and playoffs only. Source: GET /v1/club-schedule-season/{team}/{season}.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `nhl_game_id` | BIGINT | no | PK | NHL game id (e.g. 2025020001: season 2025, type 02, game 0001). PK. |
| `date` | DATE | no |  | Game calendar date (Eastern Time) from the API gameDate. |
| `start_time_utc` | TIMESTAMP | yes |  | Scheduled puck drop in UTC (API startTimeUTC). Used for lineup locking. NULL only for rows synced before this column existed; backfilled by re-running the schedule sync. |
| `season` | INTEGER | no |  | NHL season as start+end year int (e.g. 20252026). |
| `game_type` | INTEGER | no |  | NHL game type: 2 = regular season, 3 = playoffs. Preseason (1) is intentionally not synced. |
| `home_team_abbrev` | VARCHAR(3) | no | FK nhl_teams.abbrev | FK nhl_teams.abbrev, home side. |
| `away_team_abbrev` | VARCHAR(3) | no | FK nhl_teams.abbrev | FK nhl_teams.abbrev, away side. |
| `game_state` | VARCHAR(16) | no |  | NHL game state. OFF/FINAL = completed (only these get game logs); FUT = scheduled. Other live states may appear transiently. |
| `home_score` | INTEGER | yes |  | Final home goals. NULL until the game is played. |
| `away_score` | INTEGER | yes |  | Final away goals. NULL until the game is played. |

Indexes: `ix_nhl_games_date` (date), `ix_nhl_games_season` (season)

## `skater_game_logs`

Per-skater per-game stats. Primary source: GET /v1/gamecenter/{gameId}/boxscore (one row per dressed skater); ppp and shp filled afterwards from GET /v1/player/{id}/game-log (the boxscore carries neither). See docs/architecture.md 'Game logs merge two NHL API sources'.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `player_id` | BIGINT | no | PK, FK players.nhl_id | FK players.nhl_id. Part of PK. |
| `game_id` | BIGINT | no | PK, FK nhl_games.nhl_game_id | FK nhl_games.nhl_game_id. Part of PK. |
| `team_abbrev` | VARCHAR(3) | yes | FK nhl_teams.abbrev | FK nhl_teams.abbrev. The team this player dressed for IN THIS GAME, taken from the side of the boxscore they were listed under. Distinct from players.team_abbrev, which is a single current value and is therefore wrong for past seasons and for anyone who was traded. The model needs this to know the opponent and whether the game was played at home. NULL only for rows synced before this column existed. |
| `position` | VARCHAR(2) | yes |  | Position actually skated this game (C/LW/RW/D), from the boxscore — game-specific, not the roster's static primary_position (players.position). Drives multi-position eligibility. NULL for rows synced before this column existed; backfilled by re-running the game-log sync. NHL L/R mapped to LW/RW. |
| `goals` | INTEGER | no |  | Goals scored. |
| `assists` | INTEGER | no |  | Assists (primary + secondary). |
| `points` | INTEGER | no |  | Points (goals + assists). |
| `plus_minus` | INTEGER | no |  | Plus/minus: even-strength and shorthanded goal differential while on ice. |
| `pim` | INTEGER | no |  | Penalty minutes. |
| `ppp` | INTEGER | yes |  | Power-play points (PP goals + PP assists). NULL until the player-stats sync fills it from the per-player game-log endpoint; stays NULL for dressed-but-did-not-play rows (0:00 TOI), which that endpoint omits. |
| `shp` | INTEGER | yes |  | Shorthanded points (SH goals + SH assists). Same source and NULL semantics as ppp: the boxscore does not carry it, so the player-stats sync fills it from the per-player game-log endpoint. Scored by the Yahoo league, so it is not optional here. |
| `sog` | INTEGER | no |  | Shots on goal. |
| `hits` | INTEGER | no |  | Hits delivered. |
| `blocks` | INTEGER | no |  | Shots blocked. |
| `toi_seconds` | INTEGER | no |  | Time on ice in seconds (API 'MM:SS' parsed at ingest). |

Indexes: `ix_skater_game_logs_game` (game_id)

## `goalie_game_logs`

Per-goalie per-game stats; only goalies with >0 seconds played get a row. Primary source: GET /v1/gamecenter/{gameId}/boxscore; save_pctg and shutouts filled afterwards from GET /v1/player/{id}/game-log.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `player_id` | BIGINT | no | PK, FK players.nhl_id | FK players.nhl_id. Part of PK. |
| `game_id` | BIGINT | no | PK, FK nhl_games.nhl_game_id | FK nhl_games.nhl_game_id. Part of PK. |
| `team_abbrev` | VARCHAR(3) | yes | FK nhl_teams.abbrev | FK nhl_teams.abbrev. The team this player dressed for IN THIS GAME, taken from the side of the boxscore they were listed under. Distinct from players.team_abbrev, which is a single current value and is therefore wrong for past seasons and for anyone who was traded. The model needs this to know the opponent and whether the game was played at home. NULL only for rows synced before this column existed. |
| `decision` | VARCHAR(1) | yes |  | Game decision: W = win, L = regulation loss, O = overtime/shootout loss. NULL when the goalie played but was not charged with a decision. |
| `wins` | INTEGER | no |  | 0 or 1; derived at ingest from decision = 'W'. |
| `goals_against` | INTEGER | no |  | Goals allowed. |
| `saves` | INTEGER | no |  | Saves made. |
| `shots_against` | INTEGER | no |  | Shots faced (saves + goals). |
| `save_pctg` | DOUBLE PRECISION | yes |  | Save percentage as a 0-1 fraction (e.g. 0.925). NULL until the player-stats sync fills it from the per-player game-log endpoint; the API may also omit it when no shots were faced. |
| `shutouts` | INTEGER | yes |  | 0 or 1, per NHL shutout rules (sole goalie, zero goals against, win). NULL until the player-stats sync fills it. |
| `toi_seconds` | INTEGER | no |  | Time on ice in seconds (API 'MM:SS' parsed at ingest). |
| `started` | BOOLEAN | no |  | True if this goalie started the game. |

Indexes: `ix_goalie_game_logs_game` (game_id)

## `nhl_plays`

NHL play-by-play events (analytics dataset). Source: GET /v1/gamecenter/{gameId}/play-by-play. Common fields promoted to columns, per-event fields in JSONB details.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `game_id` | BIGINT | no | PK, FK nhl_games.nhl_game_id | FK nhl_games.nhl_game_id. Part of PK. |
| `event_id` | INTEGER | no | PK | NHL eventId, unique within a game. Part of PK. |
| `sort_order` | INTEGER | no |  | API sortOrder; the canonical chronological order of events in the game. |
| `period_number` | INTEGER | no |  | Period number (1-3 regulation, 4+ OT, shootout per periodType). |
| `period_type` | VARCHAR(8) | no |  | Period type: REG, OT, or SO (periodDescriptor.periodType). |
| `time_in_period` | VARCHAR(5) | no |  | Elapsed time in the period, 'MM:SS' (as the API reports it). |
| `type_code` | INTEGER | no |  | NHL numeric event type code. |
| `type_desc_key` | VARCHAR(32) | no |  | Event type key: faceoff, shot-on-goal, missed-shot, blocked-shot, goal, hit, giveaway, takeaway, penalty, stoppage, period-start/end, game-end, etc. |
| `situation_code` | VARCHAR(4) | yes |  | Strength/situation code, e.g. '1551' = 5v5 (away goalie, away skaters, home skaters, home goalie). NULL when the API omits it. |
| `home_team_defending_side` | VARCHAR(5) | yes |  | 'left' or 'right' — the side the home team defends this event, for normalizing x/y coordinates. NULL on events without it. |
| `x_coord` | INTEGER | yes |  | Event x coordinate on the rink (approx -100..100; details.xCoord). NULL for events without a location (stoppages, period boundaries). |
| `y_coord` | INTEGER | yes |  | Event y coordinate (approx -42..42; details.yCoord). NULL when absent. |
| `zone_code` | VARCHAR(1) | yes |  | Zone of the event relative to the event-owner team: O(ffensive), D(efensive), N(eutral). NULL when absent. |
| `event_owner_team_id` | INTEGER | yes |  | NHL numeric team id that 'owns' the event (shooter's/hitter's team, etc.; details.eventOwnerTeamId). NULL when absent. |
| `details` | JSONB | no |  | Raw API `details` object for the event (camelCase keys preserved): the per-event-type fields not promoted above — role player ids (shootingPlayerId, scoringPlayerId, assist{1,2}PlayerId, winning/losingPlayerId, hitting/hitteePlayerId, committedBy/drawnByPlayerId), goalieInNetId, shotType, running scores/SOG, clip urls. Empty object for events with no details. |

Indexes: `ix_nhl_plays_details` (details), `ix_nhl_plays_type` (type_desc_key)

## `player_injuries`

Current player injuries from ESPN's unofficial feed (no official NHL source). One row per injured player; refreshed wholesale each sync (recovered players are removed). Matched to players by normalized name + team.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `player_id` | BIGINT | no | PK, FK players.nhl_id | FK players.nhl_id. PK — a player has at most one current injury. |
| `status` | VARCHAR(32) | no |  | ESPN status: 'Out', 'Injured Reserve', 'Day-To-Day', 'Suspension', etc. |
| `injury_type` | VARCHAR(64) | yes |  | Body part / injury type (ESPN details.type, e.g. 'Knee'). NULL if absent. |
| `detail` | VARCHAR(128) | yes |  | Extra detail (ESPN details.detail, e.g. 'Surgery'). NULL if absent. |
| `return_date` | VARCHAR(32) | yes |  | Estimated return (ESPN details.returnDate), stored as given (a date or descriptive text). NULL if absent. |
| `comment` | VARCHAR(255) | yes |  | Short human note (ESPN shortComment). NULL if absent. |
| `espn_updated` | VARCHAR(32) | yes |  | ESPN's last-updated timestamp for the injury (raw string). NULL if absent. |
| `synced_at` | TIMESTAMP | no |  | When this row was last refreshed from ESPN, UTC (set each sync). |

## `league_settings`

One row per Yahoo league season. Source: GET /fantasy/v2/league/{league_key}/settings (fantasysports.yahooapis.com).

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `league_key` | VARCHAR(32) | no | PK | Yahoo league key, '{game_key}.l.{league_id}' (e.g. '465.l.12345'). PK. The game key changes every season, so this is season-unique. |
| `league_id` | VARCHAR(16) | no |  | Numeric league id from the league URL, without the game key. |
| `season` | INTEGER | no |  | NHL season start year as Yahoo reports it (e.g. 2026 for 2026-27). |
| `name` | VARCHAR(128) | no |  | League display name. |
| `num_teams` | INTEGER | no |  | Number of fantasy teams. |
| `scoring_type` | VARCHAR(16) | no |  | Yahoo scoring type: 'headpoint' = head-to-head points, 'head' = head-to-head categories, 'point' = season points, 'roto'. This project assumes a points-based type; a category league would need different value logic downstream. |
| `raw` | JSONB | no |  | Full unmodified settings payload from Yahoo, so a later question about a setting we did not normalize can be answered without a re-sync. |
| `synced_at` | TIMESTAMP | no |  | When this row was last read from Yahoo, UTC. |

## `league_stat_categories`

Scoring categories and point modifiers for a league, from Yahoo's settings stat_categories + stat_modifiers. Drives hockey/scoring/.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `league_key` | VARCHAR(32) | no | PK, FK league_settings.league_key | FK league_settings. |
| `stat_id` | INTEGER | no | PK | Yahoo's numeric stat id (e.g. 1 = Goals). Part of PK. |
| `name` | VARCHAR(64) | no |  | Yahoo's full stat name, e.g. 'Power Play Points'. |
| `display_name` | VARCHAR(16) | no |  | Yahoo's abbreviation, e.g. 'PPP', 'SHP', 'HIT', 'BLK'. |
| `position_type` | VARCHAR(1) | no |  | 'P' = skater stat, 'G' = goalie stat. |
| `modifier` | DOUBLE PRECISION | no |  | Fantasy points per unit of this stat. Can be negative (goals against is -1.5 in this league). |
| `source_column` | VARCHAR(32) | yes |  | Warehouse column this category reads, as 'table.column' (e.g. 'skater_game_logs.hits'). NULL means the warehouse cannot serve this category; hockey/scoring/ raises at config-load time rather than silently scoring it as zero. |

## `league_roster_positions`

Roster slots and counts for a league (C x2, D x4, BN x5, ...), from Yahoo settings roster_positions. Used by the draft export to size the pool.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `league_key` | VARCHAR(32) | no | PK, FK league_settings.league_key | FK league_settings. |
| `position` | VARCHAR(8) | no | PK | Slot code: C, LW, RW, D, G, BN, IR+, and flex codes like Util. Part of PK. |
| `count` | INTEGER | no |  | How many of this slot each roster has. |
| `is_starting` | BOOLEAN | no |  | True for slots that score (C/LW/RW/D/G); False for BN and IR+. |

## `player_crosswalk`

Yahoo <-> NHL player id mapping, built by normalized-name matching. Rows with nhl_id NULL are deliberate misses kept for review; they are never guessed. See hockey/yahoo/crosswalk.py.

| Column | Type | Null | Key | Meaning |
|---|---|---|---|---|
| `yahoo_player_id` | INTEGER | no | PK | Yahoo's numeric player id. PK. |
| `nhl_id` | BIGINT | yes | FK players.nhl_id | FK players.nhl_id. NULL when the name matched nothing or matched several players that the team tiebreak could not separate. |
| `yahoo_name` | VARCHAR(128) | no |  | Full name exactly as Yahoo spells it, for review of misses. |
| `yahoo_team` | VARCHAR(8) | yes |  | Yahoo's team abbreviation, used as the tiebreak. NULL if absent. |
| `yahoo_positions` | VARCHAR(32) | no |  | Comma-separated Yahoo eligible positions (e.g. 'C,LW'). Yahoo is the authority on eligibility for this league, not the NHL roster position. |
| `match_method` | VARCHAR(24) | no |  | How the row was decided: 'exact' (one name match), 'team_tiebreak' (several matches, one on the right team), 'ambiguous' (several, none separable) or 'unmatched' (no name match). |
| `synced_at` | TIMESTAMP | no |  | When this row was last refreshed from Yahoo, UTC. |
