import random
import base64
from typing import List, Tuple, Set
from itertools import product
from collections import deque

def generate_next_sequence(input_str: str, symbol_set: Set[str]) -> str:
    """
    Generate the next sequence by applying a one-token shift to the input.
    This is the core pattern transformation that simulates how tokens move
    through a context window:
    
    For example, with symbol_set = {'a', 'b', 'c', 'd'}:
    "a" -> "a"            (single token maps to itself)
    "ab" -> "ba"          (shift tokens)
    "abc" -> "bca"        (shift tokens)
    "abcd" -> "bcda"      (shift tokens)
    "abcda" -> "bcdaa"    (shift tokens)
    
    Args:
        input_str: Input sequence
        symbol_set: Set of valid symbols/tokens
        
    Returns:
        The next sequence in the pattern
    """
    # For empty strings or None, return empty string
    if not input_str:
        return ""
    
    # For single tokens, return identity
    if len(input_str) == 1:
        return input_str
    
    # For homogeneous sequences (all same token), return the same sequence
    if len(set(input_str)) == 1:
        return input_str
    
    # For multi-token sequences, apply one-token shift
    # Remove first token and append it to the end
    return input_str[1:] + input_str[0]

def generate_all_patterns(symbol_set: Set[str], max_length: int = 4) -> List[Tuple[str, str]]:
    """
    Generate all possible input/output patterns up to max_length,
    applying the token shift transformation.
    
    Args:
        symbol_set: Set of symbols to use (e.g., {'a', 'b', 'c', 'd'})
        max_length: Maximum length of inputs to generate
        
    Returns:
        List of (input, output) pattern tuples
    """
    patterns = []
    symbols = list(symbol_set)  # Convert set to list for product()
    
    # Generate patterns for each length 1 to max_length
    for length in range(1, max_length + 1):
        # Generate all possible combinations of tokens at this length
        for combo in product(symbols, repeat=length):
            input_str = ''.join(combo)
            
            # Generate output by applying the token shift transformation
            output_str = generate_next_sequence(input_str, symbol_set)
            
            # Add to patterns
            patterns.append((input_str, output_str))
            
    return patterns

def generate_dataset(patterns: List[Tuple[str, str]], num_samples: int, output_file: str) -> int:
    """
    Generate a dataset file with the specified number of samples.
    
    Args:
        patterns: List of (input, output) pattern tuples
        num_samples: Number of samples to generate
        output_file: Path to output file
        
    Returns:
        Number of samples generated
    """
    # Sample patterns with replacement to ensure we get exactly the requested number
    sampled_patterns = []
    
    for _ in range(num_samples):
        sampled_patterns.append(random.choice(patterns))
    
    # Shuffle the samples
    random.shuffle(sampled_patterns)
    
    # Write to file
    with open(output_file, 'w') as f:
        for input_str, output_str in sampled_patterns:
            # Convert to bytes then base64
            input_b64 = base64.b64encode(input_str.encode()).decode()
            output_b64 = base64.b64encode(output_str.encode()).decode()
            f.write(f"{input_b64}\t{output_b64}\n")
    
    return len(sampled_patterns)

def verify_patterns(patterns: List[Tuple[str, str]], expected_patterns: List[Tuple[str, str]]) -> bool:
    """
    Verify that generated patterns match expected patterns.
    
    Args:
        patterns: Generated patterns
        expected_patterns: Expected patterns
        
    Returns:
        True if all expected patterns are found, False otherwise
    """
    pattern_dict = {input_str: output_str for input_str, output_str in patterns}
    
    all_found = True
    for input_str, expected_output in expected_patterns:
        if input_str in pattern_dict:
            actual_output = pattern_dict[input_str]
            if actual_output != expected_output:
                print(f"Pattern mismatch for '{input_str}': Expected '{expected_output}', got '{actual_output}'")
                all_found = False
        else:
            print(f"Pattern not found: '{input_str}' -> '{expected_output}'")
            all_found = False
    
    return all_found

def main() -> None:
    """
    Generate training and validation datasets using the one-token shift pattern transformation.
    The generator now uses 4 tokens {'a', 'b', 'c', 'd'} instead of just 2.
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Configuration - updated to 4 active tokens
    symbol_set = {'a', 'b', 'c', 'd'}  # Extended to 4 tokens
    max_length = 4
    train_samples = 10000
    val_samples = 2000
    
    # Generate all patterns
    patterns = generate_all_patterns(symbol_set, max_length)
    
    # Expected patterns for verification (a few examples with 4 tokens)
    expected_patterns = [
        # Single token identity
        ("a", "a"),
        ("b", "b"),
        ("c", "c"),
        ("d", "d"),
        
        # 2-token shift
        ("ab", "ba"),
        ("ac", "ca"),
        ("bd", "db"),
        
        # 3-token shift
        ("abc", "bca"),
        ("acd", "cda"),
        ("bcd", "cdb"),
        
        # 4-token shift
        ("abcd", "bcda"),
        ("acdb", "cdba"),
        ("bdac", "dacb"),
        
        # Homogeneous sequences
        ("aaa", "aaa"),
        ("bbbb", "bbbb"),
        ("cccc", "cccc"),
        ("dddd", "dddd")
    ]
    
    # Verify patterns match expected outputs
    print("Verifying pattern generation...")
    verification_result = verify_patterns(patterns, expected_patterns)
    if verification_result:
        print("✅ All patterns verified correctly!")
    else:
        print("❌ Pattern verification failed!")
    
    # Count patterns by input length
    length_counts = {}
    for input_str, _ in patterns:
        length = len(input_str)
        length_counts[length] = length_counts.get(length, 0) + 1
    
    # Print statistics about patterns
    print("\nPattern counts by input length:")
    for length in range(1, max_length + 1):
        print(f"  Length {length}: {length_counts.get(length, 0)} patterns")
    print(f"  Total: {len(patterns)} patterns")
    
    # Print examples of patterns for each length
    print("\nPattern examples by length:")
    for length in range(1, max_length + 1):
        examples = [(inp, out) for inp, out in patterns if len(inp) == length][:5]  # Show up to 5 examples
        if examples:
            print(f"\n  Length {length} examples:")
            for input_str, output_str in examples:
                print(f"    '{input_str}' -> '{output_str}'")
    
    # Generate datasets with informative filenames
    output_prefix = 'data/cycle_4t_4cw'  # Updated to reflect 4 tokens
    train_file = f'{output_prefix}-train.txt'
    val_file = f'{output_prefix}-val.txt'
    
    print("\nGenerating datasets...")
    train_count = generate_dataset(patterns, train_samples, train_file)
    val_count = generate_dataset(patterns, val_samples, val_file)
    
    # Print dataset statistics
    print(f"  Training dataset: {train_count} samples (target: {train_samples})")
    print(f"  Validation dataset: {val_count} samples (target: {val_samples})")
    print(f"  Files: {train_file}, {val_file}")
    
    # Calculate the total number of possible patterns
    total_possible = sum(len(symbol_set) ** length for length in range(1, max_length + 1))
    total_generated = len(patterns)
    
    # Print pattern coverage statistics
    print(f"\nPattern coverage:")
    print(f"  Total possible patterns with 4 tokens up to length {max_length}: {total_possible}")
    print(f"  Total generated patterns: {total_generated}")
    print(f"  Coverage: {total_generated/total_possible*100:.1f}%")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
