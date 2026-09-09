"""Regenerate docs/data-dictionary.md from the live schema's COMMENT ON text.

The comments in the migrations are the source of truth; this file is derived
and must never be hand-edited, or the two drift apart and the document becomes
worse than nothing. Run after any schema change:

    python scripts/generate_data_dictionary.py
"""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import inspect, text

from hockey.db import engine

OUTPUT = Path("docs/data-dictionary.md")

TABLE_ORDER = [
    "nhl_teams",
    "players",
    "nhl_games",
    "skater_game_logs",
    "goalie_game_logs",
    "nhl_plays",
    "player_injuries",
    "league_settings",
    "league_stat_categories",
    "league_roster_positions",
    "player_crosswalk",
]


def table_comment(connection, table: str) -> str:
    return (
        connection.execute(
            # cast(... as regclass), not ::regclass - the :: cast collides
            # with SQLAlchemy's :name bind-parameter syntax.
            text("SELECT obj_description(cast(:t AS regclass), 'pg_class')"),
            {"t": table},
        ).scalar()
        or ""
    )


def main() -> None:
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    tables = [t for t in TABLE_ORDER if t in present]
    tables += sorted(present - set(TABLE_ORDER) - {"alembic_version"})

    lines = [
        "# Data dictionary",
        "",
        "**Generated** from the schema's `COMMENT ON` text by",
        "`scripts/generate_data_dictionary.py`. Do not edit by hand - change the",
        "comment in the migration and regenerate, or this drifts out of step with",
        "the database and becomes worse than no document at all.",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d')} against {len(tables)} tables.",
        "",
    ]

    with engine.connect() as connection:
        for table in tables:
            comment = table_comment(connection, table)
            pk = set(inspector.get_pk_constraint(table).get("constrained_columns") or [])
            fks = {
                column: f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                for fk in inspector.get_foreign_keys(table)
                for column in fk["constrained_columns"]
            }
            indexes = inspector.get_indexes(table)

            lines.append(f"## `{table}`")
            lines.append("")
            if comment:
                lines.append(comment)
                lines.append("")
            lines.append("| Column | Type | Null | Key | Meaning |")
            lines.append("|---|---|---|---|---|")
            for column in inspector.get_columns(table):
                name = column["name"]
                key_parts = []
                if name in pk:
                    key_parts.append("PK")
                if name in fks:
                    key_parts.append(f"FK {fks[name]}")
                meaning = (column.get("comment") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{name}` | {column['type']} | "
                    f"{'yes' if column['nullable'] else 'no'} | "
                    f"{', '.join(key_parts)} | {meaning} |"
                )
            lines.append("")
            if indexes:
                listed = ", ".join(
                    f"`{i['name']}` ({', '.join(i['column_names'])})" for i in indexes
                )
                lines.append(f"Indexes: {listed}")
                lines.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(tables)} tables)")


if __name__ == "__main__":
    main()
