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
    yahoo_redirect_uri: str = "https://localhost:8000"
    yahoo_league_id: str = ""
    yahoo_token_path: str = ".yahoo_token.json"


settings = Settings()
