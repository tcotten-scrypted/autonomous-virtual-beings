import random
import base64
import itertools

def generate_cycle_2cw_dataset(num_samples, output_file):
    patterns = [
        # Single characters (no change)
        (b'a', b'a'),
        (b'b', b'b'),
        (b'c', b'c'),
        (b'd', b'd'),

        # Same character pairs (no change)
        (b'aa', b'aa'),
        (b'bb', b'bb'),
        (b'cc', b'cc'),
        (b'dd', b'dd'),

        # Cycle for 4-character transformation
        (b'ab', b'ba'),
        (b'ac', b'ca'),
        (b'ad', b'da'),
        (b'ba', b'ab'),
        (b'bc', b'cb'),
        (b'bd', b'db'),
        (b'ca', b'ac'),
        (b'cb', b'bc'),
        (b'cd', b'dc'),
        (b'dc', b'cd'),
        (b'da', b'ad'),
        (b'ad', b'da')
    ]

    # Expand patterns to increase dataset size
    expanded_patterns = []
    for input_bytes, target_bytes in patterns:
        for _ in range(num_samples // len(patterns)):
            expanded_patterns.append((input_bytes, target_bytes))
    
    random.shuffle(expanded_patterns)

    # Write dataset
    with open(output_file, 'w') as f:
        for input_bytes, target_bytes in expanded_patterns:
            input_b64 = base64.b64encode(input_bytes).decode('utf-8')
            target_b64 = base64.b64encode(target_bytes).decode('utf-8')
            f.write(f"{input_b64}\t{target_b64}\n")

# Generate train and validation datasets
generate_cycle_2cw_dataset(10000, '../data/cycle_4t_2cw-train.txt')
generate_cycle_2cw_dataset(2000, '../data/cycle_4t_2cw-val.txt')
