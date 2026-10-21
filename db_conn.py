import os
import json
import dotenv
import psycopg2

dotenv.load_dotenv()  # Load environment variables from .env file


def get_db_connection():
	"""Connect using PGHOST, PGPORT, PGDATABASE, PGUSER, and PGPASSWORD env vars.

	Set PGDATABASE to the database name and PGPASSWORD to its password; do not
	replace the os.environ lookups with literal credentials in this file.
	"""
	return psycopg2.connect(
		host=os.getenv("PGHOST", "localhost"),
		port=os.getenv("PGPORT", "5432"),
		dbname=os.getenv("PGDATABASE", "ParakhDB"),
		user=os.getenv("PGUSER", "varunxsensei"),
		password=os.getenv("PGPASSWORD", "1234"),
	)


def test_db_connection():
	"""Run a simple query and print its result."""
	connection = get_db_connection()
	try:
		with connection.cursor() as cursor:
			cursor.execute("SELECT 1")
			print(cursor.fetchone())
	finally:
		connection.close()


def execute(query, params=None):
	"""Execute a query and return its rows as a JSON string."""
	connection = get_db_connection()
	try:
		with connection.cursor() as cursor:
			cursor.execute(query, params)
			if cursor.description is None:
				return json.dumps([])
			columns = [column.name for column in cursor.description]
			rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
			return json.dumps(rows, default=str)
	finally:
		connection.close()

		

def execute_many(query, params_list):
	"""Execute a parameterized write for each parameter set and commit it."""
	connection = get_db_connection()
	try:
		with connection.cursor() as cursor:
			cursor.executemany(query, params_list)
			row_count = cursor.rowcount
		connection.commit()
		return row_count
	finally:
		connection.close()

