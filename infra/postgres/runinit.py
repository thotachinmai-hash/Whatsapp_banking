import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

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