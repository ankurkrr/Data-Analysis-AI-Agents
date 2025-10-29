import mysql.connector
import os
import json
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "tcs_forecast"),
}

class MySQLClient:
    def __init__(self):
        db_name = MYSQL_CONFIG["database"]

        # Step 1: Connect to MySQL *without* database to ensure it exists
        temp_conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"]
        )
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        temp_conn.commit()
        temp_cursor.close()
        temp_conn.close()

        # Step 2: Connect to the actual database
        self.conn = mysql.connector.connect(**MYSQL_CONFIG)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            request_uuid VARCHAR(64) UNIQUE,
            payload JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            request_uuid VARCHAR(64),
            result_json JSON,
            tools_raw JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (request_uuid)
        )
        """)
        self.conn.commit()
        cur.close()

    def log_request(self, request_uuid: str, payload: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO requests (request_uuid, payload) VALUES (%s, %s)",
            (request_uuid, json.dumps(payload)),
        )
        self.conn.commit()
        cur.close()

    def log_result(self, request_uuid: str, result: dict, tools_raw: dict = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO results (request_uuid, result_json, tools_raw) VALUES (%s, %s, %s)",
            (request_uuid, json.dumps(result), json.dumps(tools_raw or {})),
        )
        self.conn.commit()
        cur.close()

    def get_result(self, request_uuid: str):
        cur = self.conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM results WHERE request_uuid=%s ORDER BY created_at DESC LIMIT 1",
            (request_uuid,),
        )
        r = cur.fetchone()
        cur.close()
        return r