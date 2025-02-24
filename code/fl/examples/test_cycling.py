"""
Test cycling script for the minimal Transformer with sliding window visualization.
"""

import os
import sys
import time
from typing import List, Optional
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
        Layout(name="main", ratio=1),
        Layout(name="side", ratio=2)
    )

    # Split side panel into stats and log
    layout["side"].split(
        Layout(name="stats", size=10),  # Increased height for stats panel to show all content
        Layout(name="log")
    )

    # Initialize main panel
    layout["main"].update(
        Panel(
            Text("Initializing...", style="yellow"),
            title="Generated Text",
            border_style="blue",
            box=box.ROUNDED,
            height=15  # Fixed height for scrolling
        )
    )

    return layout


def format_stats(
    total_tokens: int,
    window_sizes: List[int],
    iterations: int,
    temperature: float = 0.8,
    max_context: int = 64,
    device: Optional[torch.device] = None
) -> Text:
    """Format statistics for display."""
    avg_window = sum(window_sizes) / len(window_sizes) if window_sizes else 0

    text = Text()
    text.append("Statistics", style="bold magenta")
    text.append("\n\n")

    stats = [
        ("Iterations", f"{iterations}/1000"),
        ("Total Tokens", str(total_tokens)),
        ("Avg Window Size", f"{avg_window:.1f}"),
        ("Max Context", str(max_context)),
        ("Current Window", str(window_sizes[-1] if window_sizes else 0)),
        ("Temperature", f"{temperature:.2f}"),
        ("Device", str(device if device else 'Not initialized'))
    ]

    for label, value in stats:
        text.append(f"{label}: ", style="cyan")
        text.append(f"{value}\n", style="green")

    return text


def format_log(log_entries: List[str]) -> str:
    """Format log entries for display."""
    return "\n".join(log_entries[-10:])  # Show last 10 entries

def format_for_display(raw_text: str) -> str:
    """
    Ensures the text is safely printable by replacing control characters with '<?>'
    while preserving spaces, tabs, and newlines.

    - Avoids slow utf-8 decoding.
    - Keeps formatting responsive in Rich UI.
    - Retains whitespace characters (\t, \n, and spaces) for readability.
    """
    return ''.join(
        c if c in {' ', '\t', '\n'} or unicodedata.category(c)[0] != 'C' else '�'
        for c in raw_text)

def extract_val_loss(filename: str) -> float:
    """Extract validation loss from checkpoint filename."""
    try:
        match = re.search(r'val_loss=(\d+\.\d+)', filename)
        if match:
            return float(match.group(1))
        return float('inf')
    except (ValueError, AttributeError):
        return float('inf')


def main():
    try:
        # Find latest checkpoint
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")

        if not os.path.exists(checkpoint_path):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

            # Sort checkpoints by validation loss
            latest_checkpoint = min(checkpoints, key=extract_val_loss)
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint: {checkpoint_path}")

        # Initialize model and get device
        model = MinimalTransformer.load_from_checkpoint(checkpoint_path)
        model.eval()
        device = model.device
        
        print(f"Model initialized on device: {device}")

        # Generation parameters
        temperature = 0.8
        max_context = 64
        top_k = 128
        print(f"Generation parameters: temperature={temperature}, max_context={max_context}")

        # Initialize tracking variables
        # raw_text is used for token generation; display_text is computed for UI
        raw_text = "gm 😊"
        log_entries: List[str] = []
        window_sizes: List[int] = []
        total_tokens = len(raw_text)
        iterations = 0
        interval_tokens: List[str] = []

        # Create and configure layout
        layout = create_layout()
        console = Console()

        # Main display loop
        with Live(layout, console=console, screen=True, refresh_per_second=4) as live:
            while iterations < 1000:
                try:
                    # Use raw_text for token generation
                    input_sequence = raw_text[-(max_context - 1):] if len(raw_text) > (max_context - 1) else raw_text
                    window_sizes.append(len(input_sequence))

                    # Update statistics panel
                    layout["stats"].update(
                        Panel(
                            format_stats(
                                total_tokens=total_tokens,
                                window_sizes=window_sizes,
                                iterations=iterations,
                                temperature=temperature,
                                max_context=max_context,
                                device=device
                            ),
                            title="Statistics",
                            border_style="green",
                            box=box.ROUNDED
                        )
                    )

                    # Convert to tensor
                    input_tokens = torch.tensor(
                        [min(ord(c), 255) for c in input_sequence],
                        dtype=torch.long,
                        device=device
                    ).unsqueeze(0)

                    # Generate next token using temp and top_k=10
                    with torch.no_grad():
                        logits = model(input_tokens)      
                           
                        next_token_logits = logits[0, -1, :]
                        next_token_logits = next_token_logits / temperature

                        top_values, top_indices = torch.topk(next_token_logits, top_k)
                        top_probs = torch.softmax(top_values, dim=-1)
    
                        next_token_id = torch.multinomial(top_probs, num_samples=1)
                        next_token = top_indices[next_token_id]

                    # Convert to character (do not filter for generation)
                    next_char = chr(min(next_token.item(), 255))
                    # Append the new token to the raw text
                    raw_text += next_char
                    total_tokens += 1
                    iterations += 1
                    interval_tokens.append(next_char)

                    # Compute display text from the raw text,
                    # allowing line breaks and combining sequences
                    display_text = format_for_display(raw_text)
                    layout["main"].update(
                        Panel(
                            Text(display_text, overflow='fold', style="white"),
                            title="Generated Text",
                            border_style="blue",
                            box=box.ROUNDED,
                            height=10
                        )
                    )

                    # Update log every 10 iterations
                    if iterations % 10 == 0:
                        # Compute a display version of the latest interval tokens
                        tokens_display = format_for_display(''.join(interval_tokens))
                        # Create a lambda to map each character to its numeric (ord) value
                        to_ids = lambda seq: ', '.join(str(ord(c)) for c in seq)

                        log_entries.append(
                            f"Iterations {iterations-9}-{iterations}: '{tokens_display}' [IDs: {to_ids(interval_tokens)}]"
                        )
                        
                        interval_tokens = []
                        layout["log"].update(
                            Panel(
                                Text(format_log(log_entries), style="white"),
                                title="Log",
                                border_style="yellow",
                                box=box.ROUNDED
                            )
                        )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    log_entries.append(error_msg)
                    print(error_msg)  # Also print to console for debugging
                    time.sleep(1)  # Pause briefly on error
                    continue

            input("Press any key to exit...")

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")


if __name__ == "__main__":
    main()
