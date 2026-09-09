from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://hockey:hockey@localhost:5434/hockey_models"

    # The NHL public API. Only hockey.ingest may call it (see docs/architecture.md).
    nhl_api_base_url: str = "https://api-web.nhle.com/v1"
    # ESPN's unofficial site API - the injury source, which feeds the model's
    # availability component. No official NHL injury feed exists.
    espn_api_base_url: str = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"

    # --- Yahoo Fantasy Sports API ---
    yahoo_client_id: str = ""
    yahoo_client_secret: str = ""
    # Yahoo rejects localhost and 127.0.0.1 as redirect URIs. An app with no
    # hosted callback uses "oob", where Yahoo displays the authorization code
    # on the page rather than redirecting.
    yahoo_redirect_uri: str = "oob"
    # Yahoo issues a token with no fantasy access at all when no scope is
    # requested, and every Fantasy API call then returns 403 "This application
    # is not authorized to perform this action" - which reads like an app
    # misconfiguration but is not. Which scope is valid depends on how the app
    # was registered: an app with Read/Write permission accepts only fspt-w and
    # rejects fspt-r as an invalid scope. This project only ever issues GETs.
    yahoo_scope: str = "fspt-w"
    yahoo_league_id: str = ""
    yahoo_token_path: str = ".yahoo_token.json"


settings = Settings()
