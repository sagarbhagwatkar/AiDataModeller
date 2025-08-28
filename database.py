"""Database connectivity utilities for PostgreSQL.

Provides:
 - Pydantic settings driven configuration (env override friendly)
 - psycopg2 connection helper (lazy)
 - SQLAlchemy engine helper
 - Simple query execution helper returning rows / rowcount
 - Context manager for transactional work

Security:
 The password and other credentials should be stored in environment variables
 or a .env file. Although the user supplied a password inline, this module
 defaults to reading from environment variables and only falls back to the
 provided defaults if those are missing. Avoid logging sensitive fields.

Environment Variables (override defaults):
 POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

Example .env entries:
 POSTGRES_HOST=localhost
 POSTGRES_PORT=5432
 POSTGRES_DB=mypoc
 POSTGRES_USER=sagarbhagwatkar
 POSTGRES_PASSWORD=changeme
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union
import logging
import os

from pydantic import Field, validator
from pydantic_settings import BaseSettings

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception:  # pragma: no cover - driver optional until installed
    psycopg2 = None  # type: ignore

try:  # SQLAlchemy optional (already in requirements)
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine as SAEngine
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore
    SAEngine = Any  # type: ignore

logger = logging.getLogger(__name__)


class PostgresSettings(BaseSettings):
    """Settings for PostgreSQL connection (env overridable)."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    dbname: str = Field(default="mypoc")
    user: str = Field(default="sagarbhagwatkar")
    password: str = Field(default="Ssbsagar")  # Provided default; recommend override.
    sslmode: Optional[str] = Field(default=None)

    class Config:
        env_prefix = "POSTGRES_"  # e.g., POSTGRES_HOST
        case_sensitive = False
        extra = "ignore"

    @validator("sslmode")
    def _empty_to_none(cls, v: Optional[str]) -> Optional[str]:  # noqa: D401
        return v or None

    def dsn(self, hide_password: bool = False) -> str:
        pwd = "***" if hide_password else self.password
        base = f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.dbname}"
        if self.sslmode:
            return base + f"?sslmode={self.sslmode}"
        return base


_settings_cache: Optional[PostgresSettings] = None
_engine_cache: Any = None  # Cache for SQLAlchemy engine instance


def get_settings(force_refresh: bool = False) -> PostgresSettings:
    """Return cached Postgres settings (reload if force_refresh)."""
    global _settings_cache
    if force_refresh or _settings_cache is None:
        _settings_cache = PostgresSettings()  # auto-load from env/.env
    return _settings_cache


def get_connection():  # type: ignore[override]
    """Return a new psycopg2 connection.

    Raises:
        RuntimeError: if psycopg2 is not installed.
    """
    if psycopg2 is None:
        raise RuntimeError(
            "psycopg2 is not installed. Add 'psycopg2-binary' to requirements and pip install."
        )
    cfg = get_settings()
    conn_params: Dict[str, Any] = {
        "host": cfg.host,
        "port": cfg.port,
        "dbname": cfg.dbname,
        "user": cfg.user,
        "password": cfg.password,
    }
    if cfg.sslmode:
        conn_params["sslmode"] = cfg.sslmode
    logger.debug("Opening PostgreSQL connection to %s", cfg.dsn(hide_password=True))
    return psycopg2.connect(**conn_params)  # type: ignore[arg-type]


def get_engine(force_refresh: bool = False) -> Any:
    """Return (and cache) a SQLAlchemy Engine."""
    global _engine_cache
    if create_engine is None:
        raise RuntimeError("SQLAlchemy not available. Ensure it's installed.")
    if force_refresh or _engine_cache is None:
        cfg = get_settings()
        _engine_cache = create_engine(cfg.dsn(hide_password=False), future=True)
    return _engine_cache


def execute_query(
    sql: str,
    params: Optional[Union[Sequence[Any], Dict[str, Any]]] = None,
    fetch: bool = True,
    as_dict: bool = True,
) -> Union[List[Dict[str, Any]], List[Tuple[Any, ...]], int]:
    """Execute a SQL statement via psycopg2.

    Args:
        sql: SQL text (parameter placeholders use %s or named style depending on params)
        params: Optional params sequence or mapping.
        fetch: If True and query returns rows (SELECT), fetch them.
        as_dict: If True and fetch, return list of dict rows.

    Returns:
        List of rows (dict or tuple) or rowcount for write operations.
    """
    conn = get_connection()
    try:
        cur_factory = psycopg2.extras.RealDictCursor if (as_dict and fetch) else None
        with conn.cursor(cursor_factory=cur_factory) as cur:  # type: ignore[arg-type]
            cur.execute(sql, params)
            if fetch and cur.description:
                rows = cur.fetchall()
                return rows  # type: ignore[return-value]
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


@contextmanager
def transactional_connection() -> Generator[Any, None, None]:
    """Context manager yielding a psycopg2 connection with commit/rollback semantics."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ping() -> bool:
    """Simple connectivity check; returns True if a SELECT 1 succeeds."""
    try:
        result = execute_query("SELECT 1", fetch=True)
        return bool(result)
    except Exception as exc:  # pragma: no cover
        logger.warning("Database ping failed: %s", exc)
        return False


if __name__ == "__main__":  # Manual quick test (optional)
    cfg = get_settings()
    print("Config DSN:", cfg.dsn(hide_password=True))
    if psycopg2 is None:
        print("psycopg2 not installed; install psycopg2-binary to enable connectivity.")
    else:
        print("Ping:", ping())
