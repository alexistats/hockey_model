"""Thin Yahoo Fantasy Sports API client.

The only place in this project that talks to Yahoo, mirroring the rule that
keeps NHL calls inside hockey/ingest/. Same retry posture.

Yahoo's JSON is a mechanical translation of their XML, so it carries three
awkward shapes: collections keyed by stringified integers alongside a 'count'
key, records split across several dicts that should have been one, and lists
whose every entry is a single-key wrapper. `flatten` normalizes only the first
of those, because it is the one that is unambiguous. The other two are handled
explicitly by `merge_fragments` and `unwrap` at the call site, where the code
knows which shape it is looking at - a general rule that guesses would silently
drop repeated entries such as roster positions.
"""

import logging
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from hockey.yahoo.oauth import current_access_token

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"


class YahooShapeError(ValueError):
    """Yahoo's payload was not the shape the parser expects."""


class YahooPermissionError(RuntimeError):
    """The token is valid but carries no API permissions."""


# Yahoo answers 403 with "This application is not authorized to perform this
# action", which reads like a malformed request and is not. The token exchange
# succeeds, the token looks entirely normal, and it carries nothing - the token
# response has no scope field to reveal that. The tell is that the same token
# also 403s on Yahoo's own identity endpoint, which needs no fantasy permission
# at all, so the problem is the app rather than the scope being asked for.
PERMISSION_HELP = """Yahoo returned 403: the access token is valid but has no API permissions.

This is a setting on the Yahoo app itself, not something the code can request.
A token from an app with no permissions is refused by every Yahoo endpoint,
including the non-fantasy identity one, which is how to tell this apart from a
scope problem.

To fix it:
  1. Open https://developer.yahoo.com/apps/ and select the app.
  2. Under API Permissions, tick 'Fantasy Sports' and choose Read
     (or Read/Write).
  3. Save. Yahoo may issue a new Client ID and Secret; if it does, copy both
     into .env.
  4. Set YAHOO_SCOPE to match: fspt-r for Read, fspt-w for Read/Write.
  5. Re-authorize: python -m hockey.yahoo auth-url

If the permission cannot be added to the existing app, create a new one with
Application Type 'Installed Application' and the Fantasy Sports permission set
at creation time."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def flatten(node: Any) -> Any:
    """Turn Yahoo's numeric-keyed dicts into lists and drop their 'count' key.

    {'0': {...}, '1': {...}, 'count': 2} becomes [{...}, {...}]. Nothing else
    is restructured, so what comes out still has Yahoo's fragment lists and
    wrapper dicts in it - deliberately, because collapsing those needs to know
    what is being read.

    'count' is dropped only from a collection wrapper, never from an ordinary
    dict: Yahoo overloads the name, and a roster_position's `count` is the
    number of that slot on a roster. Stripping it everywhere would turn "four
    defence slots" into no slots at all.
    """
    if isinstance(node, dict):
        numeric = sorted((k for k in node if k.isdigit()), key=int)
        if numeric:
            return [flatten(node[k]) for k in numeric]
        if set(node) == {"count"}:
            return []  # an empty collection, not a record with a count field
        return {k: flatten(v) for k, v in node.items()}
    if isinstance(node, list):
        return [flatten(x) for x in node]
    return node


def merge_fragments(node: Any) -> dict:
    """Merge a record Yahoo split across several dicts into one dict.

    Yahoo returns a league as [{league_key, name, ...}, {settings: ...}]. The
    fragments carry disjoint keys, so merging them loses nothing - and a
    repeated key means this is a collection, not a split record, which is a
    parsing mistake worth failing on rather than silently keeping the last one.
    """
    if isinstance(node, dict):
        return node
    if not isinstance(node, list):
        raise YahooShapeError(f"expected a dict or list of dicts, got {type(node).__name__}")
    merged: dict[str, Any] = {}
    for fragment in node:
        if not isinstance(fragment, dict):
            continue  # Yahoo pads fragment lists with empty lists
        overlap = merged.keys() & fragment.keys()
        if overlap:
            raise YahooShapeError(
                f"fragments repeat the key(s) {sorted(overlap)}, so this is a collection "
                f"rather than one split record; merging it would drop entries"
            )
        merged |= fragment
    return merged


def unwrap(node: Any, key: str) -> list:
    """Strip the single-key wrapper Yahoo puts around every collection entry.

    [{'stat': {...}}, {'stat': {...}}] becomes [{...}, {...}]. A missing key is
    an error, not a skip: quietly dropping an entry would understate a league's
    categories or roster.
    """
    if node is None:
        return []
    if isinstance(node, dict):
        node = [node]
    out = []
    for entry in node:
        if not isinstance(entry, dict):
            continue  # Yahoo pads collections with empty lists
        if key not in entry:
            raise YahooShapeError(f"collection entry has no {key!r} key: {sorted(entry)}")
        out.append(entry[key])
    return out


def _as_list(node) -> list:
    """Yahoo nests a collection one or two levels deep depending on the
    endpoint. Treat a bare dict as a one-element collection so both shapes
    walk the same way."""
    if node is None:
        return []
    if isinstance(node, dict):
        return [node]
    return [x for x in node if isinstance(x, dict | list)]


class YahooFantasyClient:
    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout, follow_redirects=True)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def get(self, path: str) -> dict:
        """Raw JSON for a Fantasy API path, with the access token attached."""
        separator = "&" if "?" in path else "?"
        url = f"{path}{separator}format=json"
        start = time.monotonic()
        response = self._client.get(
            url, headers={"Authorization": f"Bearer {current_access_token()}"}
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("YAHOO GET %s -> %s (%.0f ms)", path, response.status_code, elapsed_ms)
        if response.status_code == 403:
            raise YahooPermissionError(PERMISSION_HELP)
        response.raise_for_status()
        return response.json()

    def get_flat(self, path: str) -> Any:
        """As `get`, with numeric-keyed collections turned into lists."""
        return flatten(self.get(path))["fantasy_content"]

    def current_game_key(self) -> str:
        """Yahoo's game key for the current NHL season.

        The key changes every season, so it is never hardcoded - a league key
        built from last season's game key silently addresses last season's
        league.
        """
        game = merge_fragments(self.get_flat("/game/nhl")["game"])
        return str(game["game_key"])

    def my_leagues(self) -> list[dict]:
        """Every NHL league the authenticated user is in, this season.

        This is how the league key gets confirmed rather than typed. A league
        id copied from the wrong place produces a 400 with no hint about what
        the right one was; asking Yahoo which leagues the account is actually
        in removes the guess entirely.
        """
        content = self.get_flat("/users;use_login=1/games;game_keys=nhl/leagues")
        leagues: list[dict] = []
        for user in _as_list(content.get("users")):
            user = merge_fragments(user.get("user", user))
            for game in _as_list(user.get("games")):
                game = merge_fragments(game.get("game", game))
                for league in _as_list(game.get("leagues")):
                    league = merge_fragments(league.get("league", league))
                    if "league_key" in league:
                        leagues.append(league)
        return leagues

    def league_settings(self, league_key: str) -> dict:
        """The league record merged with its settings, as one dict."""
        return merge_fragments(self.get_flat(f"/league/{league_key}/settings")["league"])

    def league_players(self, league_key: str, start: int, count: int = 25) -> list:
        """One page of the league's player pool. Yahoo caps a page at 25."""
        league = merge_fragments(
            self.get_flat(f"/league/{league_key}/players;start={start};count={count}")["league"]
        )
        return unwrap(league.get("players"), "player")

    def close(self) -> None:
        self._client.close()
