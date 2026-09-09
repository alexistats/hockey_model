"""Yahoo OAuth2: get a token once, then keep it alive.

Yahoo's flow is three-legged and needs a browser, so the one-time authorization
is a manual step (see `python -m hockey.yahoo auth-url`). Everything after that
is automatic: the refresh token does not expire on its own, so the access token
is renewed on demand and nothing has to be re-authorized before the draft.

The token file holds a live credential. It is gitignored, and nothing in this
module logs a token, a code or the client secret.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx

from hockey.config import settings

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Renew this many seconds before the access token actually expires, so a long
# request started just under the wire does not fail mid-flight.
EXPIRY_MARGIN_SECONDS = 120


class YahooAuthError(RuntimeError):
    """Authorization is missing, malformed or rejected by Yahoo."""


@dataclass
class Token:
    access_token: str
    refresh_token: str
    expires_at: float  # unix seconds

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - EXPIRY_MARGIN_SECONDS


def _token_path() -> Path:
    return Path(settings.yahoo_token_path)


def _require_credentials() -> tuple[str, str]:
    if not settings.yahoo_client_id or not settings.yahoo_client_secret:
        raise YahooAuthError(
            "YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET are not set. Copy .env.example "
            "to .env and fill them in from https://developer.yahoo.com/apps/."
        )
    return settings.yahoo_client_id, settings.yahoo_client_secret


def authorization_url() -> str:
    """The URL to open in a browser to authorize this app.

    The redirect URI must match what the app accepts exactly - Yahoo compares
    the full string, so a trailing slash is a mismatch. Yahoo also rejects
    localhost and 127.0.0.1 outright, which is why the default is "oob": the
    out-of-band flow, where Yahoo shows the code on the page instead of
    redirecting. A mismatch shows up as a redirect to /oauth2/error with
    error_description "invalid redirect uri" rather than a login page.
    """
    client_id, _ = _require_credentials()
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": settings.yahoo_redirect_uri,
            "response_type": "code",
            # Yahoo grants fantasy read access through the app's own
            # permissions rather than a scope string on this request.
            "language": "en-us",
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def _post_token(payload: dict[str, str]) -> Token:
    client_id, client_secret = _require_credentials()
    response = httpx.post(
        TOKEN_URL,
        data=payload | {"client_id": client_id, "client_secret": client_secret},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if response.status_code != 200:
        # Yahoo puts the useful part in the body; the body of a token error
        # does not contain the secret, only an error code and description.
        raise YahooAuthError(
            f"Yahoo rejected the token request ({response.status_code}): {response.text}. "
            f"The usual causes are a redirect_uri the app does not accept (Yahoo "
            f"rejects localhost; use oob), or an authorization code that has "
            f"already expired - they last about a minute."
        )
    body = response.json()
    return Token(
        access_token=body["access_token"],
        refresh_token=body["refresh_token"],
        expires_at=time.time() + float(body.get("expires_in", 3600)),
    )


def exchange_code(code: str) -> Token:
    """Trade a one-time authorization code for tokens and save them.

    The code is short-lived - roughly a minute - so this has to run promptly
    after the browser redirect.
    """
    token = _post_token(
        {
            "grant_type": "authorization_code",
            "redirect_uri": settings.yahoo_redirect_uri,
            "code": code.strip(),
        }
    )
    save_token(token)
    logger.info("authorized; token saved to %s", _token_path())
    return token


def refresh(token: Token) -> Token:
    refreshed = _post_token(
        {
            "grant_type": "refresh_token",
            "redirect_uri": settings.yahoo_redirect_uri,
            "refresh_token": token.refresh_token,
        }
    )
    save_token(refreshed)
    logger.info("access token refreshed")
    return refreshed


def save_token(token: Token) -> None:
    path = _token_path()
    path.write_text(json.dumps(asdict(token), indent=2), encoding="utf-8")


def load_token() -> Token:
    path = _token_path()
    if not path.exists():
        raise YahooAuthError(
            f"no Yahoo token at {path}. Run `python -m hockey.yahoo auth-url`, open the "
            f"URL it prints, approve, then run `python -m hockey.yahoo exchange <code>` "
            f"with the code Yahoo shows you."
        )
    return Token(**json.loads(path.read_text(encoding="utf-8")))


def current_access_token() -> str:
    """A valid access token, refreshing first if the stored one is stale."""
    token = load_token()
    if token.expired:
        token = refresh(token)
    return token.access_token
