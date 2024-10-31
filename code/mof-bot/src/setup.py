import os
import json
import argparse
from pathlib import Path
from rich.progress import Progress
from dbh import DBH
from dotenv import load_dotenv

# Define paths for setup
DB_PATH = Path(__file__).parent / "db/bin/database.sqlite"
ENV_PATH = Path(__file__).parent / ".env"
BIN_DIR = Path(__file__).parent / "db/bin"
MIGRATIONS_DIR = Path(__file__).parent / "db/migrations"

def setup_database():
    """Initializes the database if it doesn't exist."""
    dbh = DBH()
    dbh.init()

def setup_env():
    """Loads and verifies environment settings from a .env file. Creates a template if missing."""
    if not ENV_PATH.exists():
        with open(ENV_PATH, "w") as file:
            file.write("DB_PATH=./db/bin/database.sqlite\n")
        print("Created .env template.")
    load_dotenv(dotenv_path=ENV_PATH)
    print("Environment configured.")

def setup_directories():
    """Ensures necessary directories exist."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("Required directories created.")

def setup_all():
    """Checks the setup status and runs necessary setup tasks if they haven't been completed."""
    status = load_setup_status()

    tasks = [
        ("Setting up directories", setup_directories),
        ("Configuring environment", setup_env),
        ("Setting up database", setup_database),
    ]

    # Use rich progress bar to show progress for each task
    with Progress() as progress:
        task_progress = progress.add_task("Running setup", total=len(tasks))

        for task_name, task_func in tasks:
            if status.get(task_name) is not True:
                print(f"Starting task: {task_name}")
                task_func()  # Run the setup function
                status[task_name] = True  # Mark task as complete
                save_setup_status(status)  # Save status after each task
                progress.advance(task_progress)
                print(f"Completed task: {task_name}")
            else:
                print(f"Skipping completed task: {task_name}")

    print("Project setup is complete.")

def clean_all():
    """Removes files and directories created by setup tasks."""
    # Remove the database file
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Removed database file.")

    # Remove .env file
    if ENV_PATH.exists():
        ENV_PATH.unlink()
        print("Removed .env file.")

    # Optionally remove directories if empty
    if BIN_DIR.exists() and not any(BIN_DIR.iterdir()):
        BIN_DIR.rmdir()
        print("Removed empty bin directory.")
    if MIGRATIONS_DIR.exists() and not any(MIGRATIONS_DIR.iterdir()):
        MIGRATIONS_DIR.rmdir()
        print("Removed empty migrations directory.")

    print("Cleaned up project setup files.")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Project setup script")
    parser.add_argument("command", choices=["setup", "clean"], help="Setup or clean project")

    args = parser.parse_args()

    # Execute based on command
    if args.command == "setup":
        setup_all()
    elif args.command == "clean":
        clean_all()