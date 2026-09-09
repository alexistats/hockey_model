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

from hockey.config import settings as app_settings
from hockey.db import SessionLocal
from hockey.yahoo import callback as callback_mod
from hockey.yahoo import crosswalk as crosswalk_mod
from hockey.yahoo import oauth
from hockey.yahoo import settings as settings_mod
from hockey.yahoo.client import YahooFantasyClient, YahooPermissionError

logger = logging.getLogger(__name__)


def _require_league_id(arg: str | None) -> str:
    league_id = arg or app_settings.yahoo_league_id
    if not league_id:
        raise SystemExit(
            "No league id. Pass --league-id, or set YAHOO_LEAGUE_ID in .env. It is the "
            "number in your league URL: https://hockey.fantasysports.yahoo.com/hockey/<id>"
        )
    return league_id


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hockey.yahoo")
    parser.add_argument(
        "command",
        choices=[
            "login",
            "auth-url",
            "exchange",
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
