import random
import base64

def generate_cycle_2cw_dataset(num_samples, output_file):
    patterns = [
        # Single characters (no change)
        (b'a', b'a'),
        (b'b', b'b'),
        (b'c', b'c'),
        (b'd', b'd'),
        (b'e', b'e'),
        (b'f', b'f'),
        (b'g', b'g'),
        (b'h', b'h'),

        # Same character pairs (no change)
        (b'aa', b'aa'),
        (b'bb', b'bb'),
        (b'cc', b'cc'),
        (b'dd', b'dd'),
        (b'ee', b'ee'),
        (b'ff', b'ff'),
        (b'gg', b'gg'),
        (b'hh', b'hh'),

        # Cycle for 8-character transformation
        (b'ab', b'ba'),
        (b'ac', b'ca'),
        (b'ad', b'da'),
        (b'ae', b'ea'),
        (b'af', b'fa'),
        (b'ag', b'ga'),
        (b'ah', b'ha'),
        (b'ba', b'ab'),
        (b'bc', b'cb'),
        (b'bd', b'db'),
        (b'be', b'eb'),
        (b'bf', b'fb'),
        (b'bg', b'gb'),
        (b'bh', b'hb'),
        (b'ca', b'ac'),
        (b'cb', b'bc'),
        (b'cd', b'dc'),
        (b'ce', b'ec'),
        (b'cf', b'fc'),
        (b'cg', b'gc'),
        (b'ch', b'hc'),
        (b'da', b'ad'),
        (b'db', b'bd'),
        (b'dc', b'cd'),
        (b'de', b'ed'),
        (b'df', b'fd'),
        (b'dg', b'gd'),
        (b'dh', b'hd'),
        (b'ea', b'ae'),
        (b'eb', b'be'),
        (b'ec', b'ce'),
        (b'ed', b'de'),
        (b'ef', b'fe'),
        (b'eg', b'ge'),
        (b'eh', b'he'),
        (b'fa', b'af'),
        (b'fb', b'bf'),
        (b'fc', b'cf'),
        (b'fd', b'df'),
        (b'fe', b'ef'),
        (b'fg', b'gf'),
        (b'fh', b'hf'),
        (b'ga', b'ag'),
        (b'gb', b'bg'),
        (b'gc', b'cg'),
        (b'gd', b'dg'),
        (b'ge', b'eg'),
        (b'gf', b'fg'),
        (b'gh', b'hg'),
        (b'ha', b'ah'),
        (b'hb', b'bh'),
        (b'hc', b'ch'),
        (b'hd', b'dh'),
        (b'he', b'eh'),
        (b'hf', b'fh'),
        (b'hg', b'gh'),
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
generate_cycle_2cw_dataset(10000, '../data/cycle_8t_2cw-train.txt')
generate_cycle_2cw_dataset(2000, '../data/cycle_8t_2cw-val.txt')
