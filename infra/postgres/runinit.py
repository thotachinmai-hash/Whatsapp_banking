import psycopg2
import os

DATABASE_URL = "postgresql://whastapp_banking_db_user:zk958zpvzk6glUkfavq0oHXtTLitoJ42@dpg-d9pdhiu1egvs73f88vj0-a.oregon-postgres.render.com/whastapp_banking_db"
with open('C:\\Users\\91733\\OneDrive\\Documents\\new_feature_workflow\\whatsapp_banking_wa\\infra\\postgres\\init.sql', 'r') as f:
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