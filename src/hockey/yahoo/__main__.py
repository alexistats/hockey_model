"""CLI: python -m hockey.yahoo <command>

One-time setup, in order:

  1. python -m hockey.yahoo auth-url        prints a URL to open in a browser
  2. python -m hockey.yahoo exchange <code> the code from the redirect address bar
  3. python -m hockey.yahoo settings        reads the league's scoring config
  4. python -m hockey.yahoo crosswalk       maps Yahoo players to NHL ids

Steps 1 and 2 are needed once. The refresh token does not expire on its own, so
everything afterwards runs unattended.
"""

import argparse
import json
import logging
from pathlib import Path

import httpx

from hockey.config import settings as app_settings
from hockey.db import SessionLocal
from hockey.yahoo import callback as callback_mod
from hockey.yahoo import crosswalk as crosswalk_mod
from hockey.yahoo import oauth
from hockey.yahoo import settings as settings_mod
from hockey.yahoo.client import BASE_URL, YahooFantasyClient, YahooPermissionError

logger = logging.getLogger(__name__)


def _require_league_id(arg: str | None) -> str:
    league_id = arg or app_settings.yahoo_league_id
    if not league_id:
        raise SystemExit(
            "No league id. Pass --league-id, or set YAHOO_LEAGUE_ID in .env. It is the "
            "number in your league URL: https://hockey.fantasysports.yahoo.com/hockey/<id>"
        )
    return league_id


# Yahoo answers 403 for two unrelated failures, and the message is identical.
# One call separates them: Yahoo's own identity endpoint is not part of the
# Fantasy API and does not need the Fantasy permission. If identity works and
# fantasy does not, the app is missing its Fantasy Sports permission. If even
# identity refuses, the token carries nothing at all, which is a redirect-URI
# or app-registration problem and no amount of re-authorizing will fix it until
# the app itself is corrected.
IDENTITY_URL = "https://api.login.yahoo.com/openid/v1/userinfo"
PROBES = (
    ("identity", IDENTITY_URL),
    ("fantasy, public data", f"{BASE_URL}/game/nhl?format=json"),
    ("fantasy, this account", f"{BASE_URL}/users;use_login=1/games?format=json"),
)


def _doctor() -> None:
    print()
    print(f"redirect uri : {app_settings.yahoo_redirect_uri}")
    print(f"scope        : {app_settings.yahoo_scope}")
    print(f"token file   : {app_settings.yahoo_token_path}")
    print()

    # The failure this catches: change the app in .env, skip `login`, and every
    # probe below tests a token the *old* app issued. All three refuse it, and
    # the diagnosis lands on the new app, which was never contacted. The stored
    # client id settles it outright; for a token written before that field
    # existed, .env being the newer file is the same warning with less
    # certainty. Either way this is a stop, not a note - the probes are
    # worthless until it is resolved.
    token_file = Path(app_settings.yahoo_token_path)
    env_file = Path(".env")
    try:
        stale = oauth.stale_credentials(oauth.load_token())
    except oauth.YahooAuthError as exc:
        raise SystemExit(f"No usable token: {exc}") from None
    older = (
        token_file.exists()
        and env_file.exists()
        and token_file.stat().st_mtime < env_file.stat().st_mtime
    )
    if stale or older:
        why = (
            "its client id does not match YAHOO_CLIENT_ID"
            if stale
            else "it may predate the app now configured"
        )
        # Printed rather than raised with a message, so it lands on stdout in
        # order with the header above instead of interleaving from stderr.
        print("The saved token is older than the credentials it would be tested with.")
        print()
        print(f"  {token_file} was written before .env was last changed, and")
        print(f"  {why}.")
        print()
        print("Refreshing it keeps whatever the issuing app was granted, so every probe")
        print("below would fail for the old app and tell you nothing about the new one.")
        print()
        print("Authorize first, then run doctor again:")
        print()
        print("    python -m hockey.yahoo login")
        print()
        raise SystemExit(1)

    try:
        token = oauth.current_access_token()
    except oauth.YahooAuthError as exc:
        raise SystemExit(f"No usable token: {exc}") from None
    print("access token : refreshed and usable")
    print()

    results = {}
    with httpx.Client(timeout=20) as http:
        for label, url in PROBES:
            try:
                status = http.get(url, headers={"Authorization": f"Bearer {token}"}).status_code
            except httpx.HTTPError as exc:
                print(f"  {label:22} network error: {exc}")
                return
            results[label] = status
            print(f"  {label:22} {status}")
    print()

    identity_ok = results["identity"] < 400
    fantasy_ok = results["fantasy, public data"] < 400

    if identity_ok and fantasy_ok:
        print("Both work. Yahoo is reachable and the token carries Fantasy permissions.")
    elif identity_ok and not fantasy_ok:
        print(
            "The token is real - it reads your Yahoo profile - but Fantasy refuses it.\n"
            "That is a missing API permission on the app itself.\n\n"
            "  Open https://developer.yahoo.com/apps/, pick this app, and under\n"
            "  API Permissions tick 'Fantasy Sports' with Read or Read/Write.\n"
            "  Then re-authorize: python -m hockey.yahoo login\n\n"
            "Permissions are fixed when a token is issued, so the current token\n"
            "cannot be upgraded by refreshing it."
        )
    else:
        print(
            "Even Yahoo's own identity endpoint refuses this token, and that endpoint\n"
            "is not part of the Fantasy API. So the token carries no permissions at all.\n\n"
            "Rule out the two cheap causes before touching the app.\n\n"
            "  1. A newly created app issues tokens before its permissions are live,\n"
            "     and those tokens are refused everywhere. If the app is minutes old,\n"
            "     wait a few and run `login` again - this resolved itself here.\n\n"
            "  2. The requested scope may not be one the app has. Narrow the request\n"
            f"     to what it already holds ({app_settings.yahoo_scope} is asked for):\n\n"
            "         python -m hockey.yahoo login --no-scope\n\n"
            "     Identity answering 200 afterwards means the registration is sound.\n\n"
            "If neither helps, check the app registration itself.\n\n"
            "  On https://developer.yahoo.com/apps/, check in this order:\n"
            f"  1. Redirect URI(s) contains exactly {app_settings.yahoo_redirect_uri}\n"
            "     - character for character, port included, no trailing slash.\n"
            "  2. Application Type is 'Web Application'. An 'Installed Application'\n"
            "     has no client secret and will not work with this flow.\n"
            "  3. OpenID Connect Permissions includes at least 'Profile'.\n"
            "  4. API Permissions includes 'Fantasy Sports'.\n\n"
            "If the app looks right on all four, the fastest fix is to create a new\n"
            "app rather than edit this one - Yahoo does not always re-issue\n"
            "permissions on an edited app. Put the new id and secret in .env, then\n"
            "run: python -m hockey.yahoo login"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.yahoo")
    parser.add_argument(
        "command",
        choices=[
            "login",
            "auth-url",
            "exchange",
            "doctor",
            "game-key",
            "my-leagues",
            "settings",
            "crosswalk",
            "misses",
            "dump",
        ],
    )
    parser.add_argument("code", nargs="?", help="authorization code, for `exchange`")
    parser.add_argument("--league-id", default=None)
    parser.add_argument(
        "--game-key",
        default=None,
        help="override Yahoo's NHL game key; defaults to the current season's",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "auth-url":
        print()
        print("1. Open this URL in a browser and approve access:")
        print()
        print(f"   {oauth.authorization_url()}")
        print()
        if app_settings.yahoo_redirect_uri == "oob":
            print("2. Yahoo will show the authorization code on the page itself.")
            print("   Copy it. Nothing redirects anywhere.")
        else:
            print("2. Yahoo redirects to:")
            print(f"   {app_settings.yahoo_redirect_uri}?code=SOMETHING")
            print("   The page may not load; that is fine.")
            print("   Copy the value of `code` out of the address bar.")
        print()
        print("3. Run, within about a minute (the code expires quickly):")
        print("   python -m hockey.yahoo exchange <code>")
        print()
        return

    if args.command == "login":
        # The whole authorization in one step: open the browser, catch the
        # redirect, exchange the code. The code lives about a minute, which is
        # not long to be transcribing it out of a browser error page.
        try:
            code = callback_mod.capture_code(
                app_settings.yahoo_redirect_uri, oauth.authorization_url()
            )
        except callback_mod.CallbackError as exc:
            raise SystemExit(str(exc)) from None
        oauth.exchange_code(code)
        print()
        print("Authorized. Token saved; you will not need to do this again.")
        return

    if args.command == "exchange":
        if not args.code:
            raise SystemExit("usage: python -m hockey.yahoo exchange <code>")
        oauth.exchange_code(args.code)
        print("Authorized. Token saved; you will not need to do this again.")
        return

    if args.command == "doctor":
        _doctor()
        return

    client = YahooFantasyClient()
    try:
        # A permissions problem is something for the reader to go and fix, not
        # a stack trace to decode, so it prints as guidance and stops.
        if args.command == "game-key":
            print(client.current_game_key())
            return

        if args.command == "my-leagues":
            leagues = client.my_leagues()
            if not leagues:
                print(
                    "Yahoo lists no NHL leagues for this account this season. If the "
                    "draft has not happened yet the league should still appear, so "
                    "check you authorized with the right Yahoo account."
                )
                return
            print(f"{len(leagues)} NHL league(s) on this account:")
            print()
            for league in leagues:
                print(f"  league_id  {league.get('league_id')}")
                print(f"  league_key {league.get('league_key')}")
                print(f"  name       {league.get('name')}")
                print(f"  teams      {league.get('num_teams')}   season {league.get('season')}")
                print()
            return

        if args.command == "dump":
            # Saves the raw payloads so a parsing problem can be diagnosed
            # against exactly what Yahoo sent, without another round trip.
            league_id = _require_league_id(args.league_id)
            game_key = args.game_key or client.current_game_key()
            league_key = settings_mod.build_league_key(game_key, league_id)
            out = Path("artifacts/yahoo_raw")
            out.mkdir(parents=True, exist_ok=True)
            for name, path in (
                ("game", "/game/nhl"),
                ("settings", f"/league/{league_key}/settings"),
                ("players", f"/league/{league_key}/players;start=0;count=25"),
            ):
                target = out / f"{name}.json"
                target.write_text(json.dumps(client.get(path), indent=2), encoding="utf-8")
                print(f"wrote {target}")
            return

        with SessionLocal() as session:
            if args.command == "settings":
                league_id = _require_league_id(args.league_id)
                league_key = settings_mod.sync_league_settings(
                    session, client, league_id, args.game_key
                )
                print(f"League config stored for {league_key}.")
                return

            if args.command == "crosswalk":
                league_id = _require_league_id(args.league_id)
                game_key = args.game_key or client.current_game_key()
                league_key = settings_mod.build_league_key(game_key, league_id)
                counts = crosswalk_mod.sync_crosswalk(session, client, league_key)
                total = sum(counts.values())
                matched = counts["exact"] + counts["team_tiebreak"]
                print(f"{matched}/{total} Yahoo players matched to NHL ids.")
                if total - matched:
                    print(
                        f"{total - matched} unresolved. Review them with: "
                        f"python -m hockey.yahoo misses"
                    )
                return

            if args.command == "misses":
                misses = crosswalk_mod.unmatched_report(session)
                if not misses:
                    print("No unmatched players.")
                    return
                print(f"{len(misses)} unmatched Yahoo players:\n")
                for name, team, method in misses:
                    print(f"  {method:14s} {name}  ({team or 'no team'})")
                return
    except YahooPermissionError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        client.close()


if __name__ == "__main__":
    main()
