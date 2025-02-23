"""
Test cycling script for the minimal Transformer with sliding window visualization.
"""

import os
import sys
import time
from typing import List, Tuple
import unicodedata
import re

# Add project root to Python path to allow local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

from fluctlight.model import MinimalTransformer

def create_layout() -> Layout:
    """Create the layout for the UI."""
    layout = Layout(name="root")

    # Split into main and side panel
    layout.split(
        Layout(name="main", ratio=2),
        Layout(name="side", ratio=1)
    )

    # Split side panel into stats and log
    layout["side"].split(
        Layout(name="stats"),
        Layout(name="log")
    )

    return layout

def format_stats(
    total_tokens: int,
    window_sizes: List[int],
    iterations: int
) -> str:
    """Format statistics for display."""
    avg_window = sum(window_sizes) / len(window_sizes) if window_sizes else 0
    return (
        f"[bold]Statistics[/bold]\n\n"
        f"Iterations: {iterations}/1000\n"
        f"Total Tokens: {total_tokens}\n"
        f"Avg Window Size: {avg_window:.1f}\n"
        f"Max Window: 64\n"
        f"Current Window: {window_sizes[-1] if window_sizes else 0}"
    )

def format_log(log_entries: List[str]) -> str:
    """Format log entries for display."""
    return "\n".join(log_entries[-10:])  # Show last 10 entries

def is_printable(char: str) -> bool:
    """Check if a character is printable."""
    if not char:
        return False
    return unicodedata.category(char)[0] not in {'C', 'Z'}

def extract_val_loss(filename: str) -> float:
    """
    Extract validation loss from checkpoint filename, handling version suffixes.

    Example filenames:
    - transformer-epoch=99-val_loss=1.62.ckpt
    - transformer-epoch=97-val_loss=1.64-v1.ckpt
    """
    try:
        # Extract the value between val_loss= and the next non-digit character
        match = re.search(r'val_loss=(\d+\.\d+)', filename)
        if match:
            return float(match.group(1))
        return float('inf')  # Return infinity for invalid files
    except (ValueError, AttributeError):
        return float('inf')  # Return infinity for invalid files

def main():
    try:
        # Find latest checkpoint
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

        # Sort checkpoints by validation loss, handling version suffixes
        latest_checkpoint = min(checkpoints, key=extract_val_loss)
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint: {checkpoint_path}")

        # Initialize model from checkpoint
        model = MinimalTransformer.load_from_checkpoint(checkpoint_path)
        model.eval()

        # Initialize tracking variables
        current_text = "Hello"
        log_entries: List[str] = []
        window_sizes: List[int] = []
        total_tokens = len(current_text)
        iterations = 0

        # Create and configure layout
        layout = create_layout()
        console = Console()

        # Main display loop
        with Live(layout, console=console, screen=True, refresh_per_second=4) as live:
            while iterations < 1000:
                try:
                    # Get input sequence (use last 63 tokens if longer)
                    input_sequence = current_text[-63:] if len(current_text) > 63 else current_text
                    window_sizes.append(len(input_sequence))

                    # Convert to tensor on the correct device
                    input_tokens = torch.tensor(
                        [min(ord(c), 255) for c in input_sequence],
                        dtype=torch.long,
                        device=model.device
                    ).unsqueeze(0)

                    # Generate next token
                    with torch.no_grad():
                        logits = model(input_tokens)
                        next_token_logits = logits[0, -1, :] / 0.8  # temperature=0.8
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                    # Convert to character and append
                    next_char = chr(min(next_token.item(), 255))
                    if not is_printable(next_char):
                        next_char = '�'  # Replace non-printable with replacement character

                    current_text += next_char
                    total_tokens += 1
                    iterations += 1

                    # Update log
                    if iterations % 10 == 0:  # Log every 10th iteration
                        log_entries.append(f"Iteration {iterations}: Added '{next_char}'")

                    # Update display
                    layout["main"].update(
                        Panel(
                            Text(current_text[-100:], overflow='ellipsis'),
                            title="Generated Text (Last 100 chars)",
                            border_style="blue",
                            box=box.ROUNDED
                        )
                    )

                    layout["stats"].update(
                        Panel(
                            format_stats(total_tokens, window_sizes, iterations),
                            title="Statistics",
                            border_style="green",
                            box=box.ROUNDED
                        )
                    )

                    layout["log"].update(
                        Panel(
                            format_log(log_entries),
                            title="Log",
                            border_style="yellow",
                            box=box.ROUNDED
                        )
                    )

                    # Small delay to make the display readable
                    time.sleep(0.1)

                except Exception as e:
                    log_entries.append(f"Error: {str(e)}")
                    time.sleep(1)  # Pause briefly on error
                    continue

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    main()