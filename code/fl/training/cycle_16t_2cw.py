import random
import base64
import itertools

def generate_cycle_2cw_dataset(num_samples, output_file):
    # Full alphabet for 16 tokens
    tokens = list('abcdefghijklmnop')
    
    # Generate patterns
    patterns = []
    
    # Single character preservation
    for token in tokens:
        patterns.append((bytes(token, 'utf-8'), bytes(token, 'utf-8')))
    
    # Same character pair preservation
    for token in tokens:
        patterns.append((bytes(token*2, 'utf-8'), bytes(token*2, 'utf-8')))
    
    # Cyclic transformations
    for i, token1 in enumerate(tokens):
        for j, token2 in enumerate(tokens):
            if i != j:
                patterns.append(
                    (bytes(token1 + token2, 'utf-8'), 
                     bytes(token2 + token1, 'utf-8'))
                )
    
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
generate_cycle_2cw_dataset(10000, '../data/cycle_16t_2cw-train.txt')
generate_cycle_2cw_dataset(2000, '../data/cycle_16t_2cw-val.txt')
