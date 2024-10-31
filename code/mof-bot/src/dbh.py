import os
import sqlite3

class DBH:
    def __init__(self):
        # Set paths
        self.db_path = os.path.join(os.path.dirname(__file__), "../db/bin/database.sqlite")
        self.migrations_path = os.path.join(os.path.dirname(__file__), "../db/migrations/")

    def init(self):
        """
        Initializes the database by checking if it exists.
        If it doesn't, creates it and applies the initial migration.
        Returns an SQLite connection.
        """
        # Check if database exists
        if not os.path.exists(self.db_path):
            print("Database does not exist. Initializing...")
            conn = sqlite3.connect(self.db_path)
            self._run_migrations(conn)
        else:
            # If the database exists, simply connect
            conn = sqlite3.connect(self.db_path)
            print("Database already initialized.")

        return conn

    def _run_migrations(self, conn):
        """
        Runs all SQL files in the migrations folder.
        """
        cursor = conn.cursor()

        # Find all SQL migration files and apply them in sorted order
        migration_files = sorted(f for f in os.listdir(self.migrations_path) if f.endswith(".sql"))

        for filename in migration_files:
            with open(os.path.join(self.migrations_path, filename), "r") as f:
                sql = f.read()
                cursor.executescript(sql)
                print(f"Applied migration: {filename}")

        conn.commit()