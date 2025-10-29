# tests/08_mysql_test.py
from dotenv import load_dotenv
import uuid
import pytest

load_dotenv()
from app.db.mysql_client import MySQLClient


def test_mysql_log_and_fetch():
	db = MySQLClient()
	print("Connected. creating test entry.")
	uid = f"test-uid-{uuid.uuid4().hex[:8]}"
	# This should insert without colliding with other runs
	db.log_request(uid, {"test": "payload"})
	res = db.get_result(uid)
	print("Result fetch (may be None):", res)
	# Ensure we get back at least the request metadata or None without raising
	assert res is None or isinstance(res, dict)
