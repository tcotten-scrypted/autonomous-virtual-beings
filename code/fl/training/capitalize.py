import random
import base64

# Define the allowed character set
LOWERCASE_CHARS = "abcdefgh"
UPPERCASE_CHARS = "ABCDEFGH"

def generate_random_input(size=8):
    """
    Generate a random sequence of 8 characters using 'abcdefgh' only.
    """
    return "".join(random.choices(LOWERCASE_CHARS, k=size))

def transform_target(input_str):
    """
    Convert lowercase 'abcdefgh' to uppercase 'ABCDEFGH' in the target output.
    """
    return input_str.upper()

def encode_to_base64(string_data):
    """
    Encode a string to base64 format.
    """
    return base64.b64encode(string_data.encode('utf-8')).decode('utf-8')

def main():
    # Open file for dataset storage
    with open("sample-dataset.txt", "w") as f:
        for _ in range(100_000):  # Generate 100K samples
            input_str = generate_random_input()
            target_str = transform_target(input_str)
            
            # Encode to base64 for storage
            b64_input = encode_to_base64(input_str)
            b64_target = encode_to_base64(target_str)
            
            # Write as tab-separated training pair
            f.write(f"{b64_input}\t{b64_target}\n")

if __name__ == "__main__":
    main()

