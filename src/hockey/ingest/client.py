"""Thin NHL API client. The only place in the codebase that talks to the NHL API."""

import logging
import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from hockey.config import settings

logger = logging.getLogger(__name__)

# The API returns 403 to generic/empty user agents.
HEADERS = {"User-Agent": "Mozilla/5.0 (hockey-models-research)"}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class NhlApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 15.0):
        self._client = httpx.Client(
            base_url=base_url or settings.nhl_api_base_url,
            headers=HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def get_json(self, path: str) -> dict:
        start = time.monotonic()
        response = self._client.get(path)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("GET %s -> %s (%.0f ms)", path, response.status_code, elapsed_ms)
        response.raise_for_status()
        return response.json()

    def standings_now(self) -> dict:
        return self.get_json("/standings/now")

    def standings_on(self, on_date: str) -> dict:
        """Standings as of a calendar date, 'YYYY-MM-DD'. This is the only
        endpoint that reports which franchises existed in a past season, which
        is what makes a multi-season backfill possible (see
        sync.season_team_abbrevs)."""
        return self.get_json(f"/standings/{on_date}")

    def team_roster(self, team_abbrev: str, season: int) -> dict:
        return self.get_json(f"/roster/{team_abbrev}/{season}")

    def club_schedule_season(self, team_abbrev: str, season: int) -> dict:
        return self.get_json(f"/club-schedule-season/{team_abbrev}/{season}")

    def boxscore(self, game_id: int) -> dict:
        return self.get_json(f"/gamecenter/{game_id}/boxscore")

    def play_by_play(self, game_id: int) -> dict:
        return self.get_json(f"/gamecenter/{game_id}/play-by-play")

    def player_landing(self, player_id: int) -> dict:
        """A player's bio page: birth date, height, draft details."""
        return self.get_json(f"/player/{player_id}/landing")

    def player_game_log(self, player_id: int, season: int, game_type: int) -> dict:
        return self.get_json(f"/player/{player_id}/game-log/{season}/{game_type}")

    def close(self) -> None:
        self._client.close()


class EspnApiClient:
    """Thin client for ESPN's unofficial NHL site API — the injuries source.
    Separate from NhlApiClient (different host); same retry/backoff and the
    ingest-isolation rule (docs/architecture.md) so breakage stays localized."""

    def __init__(self, base_url: str | None = None, timeout: float = 20.0):
        self._client = httpx.Client(
            base_url=base_url or settings.espn_api_base_url,
            headers=HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def get_json(self, path: str) -> dict:
        start = time.monotonic()
        response = self._client.get(path)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("ESPN GET %s -> %s (%.0f ms)", path, response.status_code, elapsed_ms)
        response.raise_for_status()
        return response.json()

    def injuries(self) -> dict:
        return self.get_json("/injuries")

    def close(self) -> None:
        self._client.close()
