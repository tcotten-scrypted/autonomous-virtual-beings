import random
import base64

def create_base64_pair(input_str, target_str):
    input_b64 = base64.b64encode(input_str.encode('utf-8')).decode('utf-8')
    target_b64 = base64.b64encode(target_str.encode('utf-8')).decode('utf-8')
    return f"{input_b64}\t{target_b64}"

def generate_dataset(num_samples, output_file):
    patterns = [
        ("a", "A"),   # Single 'a' to 'A'
        ("A", "A"),   # Single 'A' to 'A'
        ("aa", "A"),  # Double 'a' to 'A'
        ("aA", "A"),  # 'a' followed by 'A' to 'A'
        ("Aa", "A"),  # 'A' followed by 'a' to 'A'
        ("AA", "A")   # Double 'A' to 'A'
    ]
    
    # Combine all patterns
    all_patterns = patterns
    
    with open(output_file, 'w') as f:
        for _ in range(num_samples):
            # Select a random pattern
            input_str, target_str = random.choice(all_patterns)
            # Write base64 encoded pair
            f.write(create_base64_pair(input_str, target_str) + '\n')

# Generate train and validation datasets
generate_dataset(800, 'capitalize_a_2cw-train.txt')
generate_dataset(200, 'capitalize_a_2cw_val.txt')
