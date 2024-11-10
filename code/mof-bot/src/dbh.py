import os
import sqlite3
from pathlib import Path
from threading import Lock
from rich.console import Console
from rich.table import Table

class DBH:
    """
    DBH (Database Handler) is a singleton class that manages a SQLite database connection
    and handles migrations. It provides an initialized connection to the database and 
    ensures that migrations are applied upon initial setup.

    Attributes
    ----------
    db_path : str
        The file path for the SQLite database file.
    migrations_path : str
        The directory path where SQL migration files are stored.
    _instance : DBH
        The singleton instance of DBH.
    _lock : Lock
        A threading lock to ensure thread-safe singleton initialization.
    _connection : sqlite3.Connection
        The SQLite connection managed by the singleton.

    Methods
    -------
    get_instance():
        Retrieves the singleton instance of DBH.
    get_connection():
        Returns the SQLite connection, initializing it if necessary.
    _initialize():
        Checks if the database exists; if not, creates it and applies migrations.
    _run_migrations():
        Applies SQL migrations in the order they appear in the migrations directory.
    _display_table_info():
        Displays high-level table info if the database was just created.
    """

    _instance = None
    _lock = Lock()

    def __init__(self):
        if DBH._instance is not None:
            raise Exception("This class is a singleton. Use 'get_instance()' to access it.")
        
        # Define paths for the database file and migrations folder
        base_dir = Path(__file__).parent.resolve()
        
        self.db_path = base_dir / "../db/bin/database.sqlite"
        self.migrations_path = base_dir / "../db/migrations/"
        self._connection = None

    @classmethod
    def get_instance(cls):
        """
        Retrieves the singleton instance of DBH. Initializes the instance if it does not yet exist.

        Returns
        -------
        DBH
            The singleton instance of DBH.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._initialize()
        return cls._instance

    def get_connection(self):
        """
        Returns the SQLite connection managed by the singleton instance. If the connection 
        is not already established, it will be initialized.

        Returns
        -------
        sqlite3.Connection
            The SQLite connection to the database.
        """
        if self._connection is None:
            self._initialize()
        return self._connection

    def _initialize(self):
        """
        Initializes the database by checking if it exists. If it does not exist,
        creates the database file, applies initial migrations, and establishes a connection.
        Displays table info if the database was just created.
        """
        db_created = False
        if not os.path.exists(self.db_path):
            print("Database does not exist. Initializing...")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path)
            db_created = True
            self._run_migrations()
        else:
            self._connection = sqlite3.connect(self.db_path)
            print("Database already initialized.")

        # If the database was just created, display high-level table info
        if db_created:
            self._display_table_info()

    def _run_migrations(self):
        """
        Runs all SQL files in the migrations directory to apply necessary database migrations.
        Each file is executed in sorted order to maintain migration sequence.
        """
        cursor = self._connection.cursor()

        # Locate and sort SQL migration files
        migration_files = sorted(f for f in os.listdir(self.migrations_path) if f.endswith(".sql"))

        for filename in migration_files:
            with open(os.path.join(self.migrations_path, filename), "r") as file:
                sql = file.read()
                cursor.executescript(sql)
                print(f"Applied migration: {filename}")

        self._connection.commit()

    def _display_table_info(self):
        """
        Displays the names and record counts of high-level tables if the database was just created.
        Uses the `rich` library to format the output as a table.
        """
        console = Console()
        table = Table(title="Database Table Summary")

        table.add_column("Table Name", style="cyan", no_wrap=True)
        table.add_column("Record Count", style="magenta")

        cursor = self._connection.cursor()

        # Retrieve all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Query the row count for each table
        for (table_name,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table.add_row(table_name, str(count))

        console.print(table)