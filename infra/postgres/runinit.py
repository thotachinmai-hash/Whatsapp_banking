import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _load_project_env() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return

    example_path = _PROJECT_ROOT / ".env.example"
    if example_path.exists():
        load_dotenv(example_path)


_load_project_env()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to the project .env or .env.example "
        f"at {_PROJECT_ROOT} before running this script."
    )

_INIT_SQL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init.sql")
with open(_INIT_SQL_PATH, "r") as f:
    sql = f.read()
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

try:
    cursor.execute(sql)
    conn.commit()
    print("Database initialized successfully.")
except Exception as e:
    print(f"Error initializing database: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()