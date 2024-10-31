import os

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

# Default logo path
default_logo_path = os.path.join(os.path.dirname(__file__), "mof_logo.txt")

# Initialize a console object from rich
console = Console()

def load_logo(file_path=default_logo_path):
    """Load ASCII logo from a text file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Logo file not found."

def display():
    logo_content = load_logo()
    logo_text = Text(logo_content, style=Style(color="blue"))

    # Create a panel to center the ASCII logo text
    logo_panel = Panel(
        logo_text,
        border_style="blue",
        padding=(1, 2),  # Adjust padding as needed
        expand=False,
    )

    # Define the colored bars at the top and bottom
    bar = Panel(
        "",
        style="bold magenta",
        padding=(0, 0),
        height=1,
        expand=True,
    )

    # Display the splash screen
    console.print(bar)
    console.print(logo_panel, justify="center")
    console.print(bar)

# Example of calling the splash screen in your main program
if __name__ == "__main__":
    display()
