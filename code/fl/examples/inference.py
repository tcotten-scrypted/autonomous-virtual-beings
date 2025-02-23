"""
Example script for running inference with the trained Transformer model.
"""

import argparse
from pathlib import Path

import torch

from fluctlight.model import MinimalTransformer
from fluctlight.utils import generate_continuation

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Transformer")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input text to continue from")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}")
    model = MinimalTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Generate multiple samples
    print(f"\nGenerating {args.num_samples} continuations for: {args.input}")
    print("-" * 50)

    for i in range(args.num_samples):
        output = generate_continuation(
            model,
            args.input,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"\nSample {i+1}:")
        print(f"Input: {args.input}")
        print(f"Generated: {output}")
        print("-" * 30)

def interactive_mode(model_path: str):
    """Run interactive generation mode."""
    model = MinimalTransformer.load_from_checkpoint(model_path)
    model.eval()

    print("\nEntering interactive mode (Ctrl+C to exit)")
    print("Enter text and press Enter to generate continuations")
    
    try:
        while True:
            input_text = input("\nInput > ")
            if not input_text.strip():
                continue

            output = generate_continuation(
                model,
                input_text,
                max_length=100,
                temperature=0.8
            )
            print(f"Generated: {output}")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode")

if __name__ == "__main__":
    main()
