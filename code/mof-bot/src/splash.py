import os
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

# Default logo path
default_logo_path = Path(__file__).parent.resolve() / "./assets/avbeing_logo.txt"

# Initialize a console object from rich
console = Console()

def load_logo(file_path=default_logo_path):
    """Load ASCII logo from a text file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Logo file not found."

def display(version="Wildcard (v0.0.0)"):
    """
    Display the ASCII logo at the top-left corner in a Matrix green color, clear the screen,
    and append the version number at the bottom of the logo. The splash screen will stay visible
    for 5 seconds before continuing.

    Parameters
    ----------
    version : str
        The version number to display at the bottom of the logo.
    """
    # Clear the console screen
    console.clear()

    # Load and style the logo and version text
    logo_content = load_logo()
    combined_text = Text(logo_content, style=Style(color="green"))
    combined_text.append(f"\nVersion: {version}", style=Style(color="green"))
    combined_text.append(f"\nAuthor:  @cottenio // scrypted", style=Style(color="green"))

    # Print the combined logo and version text directly (no panel)
    console.print(combined_text)

    # Pause to display the splash screen for 2 seconds
    time.sleep(2)