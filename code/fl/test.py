import random
import base64
import os
from tqdm import tqdm

def generate_addition_problem():
    """Generate a random addition problem with its solution."""
    # Determine complexity of the problem
    complexity = random.choices(
        ['single', 'double', 'triple', 'large'], 
        weights=[0.4, 0.3, 0.2, 0.1], 
        k=1
    )[0]
    
    if complexity == 'single':
        # Simple single digit addition
        a = random.randint(0, 9)
        b = random.randint(0, 9)
    elif complexity == 'double':
        # Double digit addition
        a = random.randint(10, 99)
        b = random.randint(10, 99)
    elif complexity == 'triple':
        # Triple digit addition
        a = random.randint(100, 999)
        b = random.randint(100, 999)
    else:
        # Larger numbers but keeping within byte limits
        a = random.randint(1000, 9999999)
        b = random.randint(1000, 9999999)
    
    # Create the problem and solution
    problem = f"{a}+{b}="
    solution = str(a + b)
    
    # Verify byte length constraints
    if len(problem.encode('utf-8')) > 33 or len(solution.encode('utf-8')) > 33:
        # Recursively try again if too long
        return generate_addition_problem()
    
    return problem, solution

def base64_encode(text):
    """Encode text to base64."""
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def create_dataset(num_samples, output_file):
    """Create a dataset of addition problems and write to file."""
    print(f"Generating {num_samples} addition problems for {output_file}...")
    
    with open(output_file, 'w') as f:
        for _ in tqdm(range(num_samples)):
            problem, solution = generate_addition_problem()
            
            # Base64 encode both problem and solution
            encoded_problem = base64_encode(problem)
            encoded_solution = base64_encode(solution)
            
            # Write to file in the format: encoded_problem\tencoded_solution
            f.write(f"{encoded_problem}\t{encoded_solution}\n")
    
    print(f"Dataset saved to {output_file}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname("data/"), exist_ok=True)
    
    # Generate training set (50,000 samples)
    create_dataset(50000, "train.test.txt")
    
    # Generate validation set (10,000 samples)
    create_dataset(10000, "val.test.txt")
    
    # Verify files
    train_size = os.path.getsize("train.test.txt") / (1024 * 1024)  # Size in MB
    val_size = os.path.getsize("val.test.txt") / (1024 * 1024)  # Size in MB
    
    print(f"\nTraining set: {train_size:.2f} MB")
    print(f"Validation set: {val_size:.2f} MB")
    
    # Sample check of a few examples
    print("\nSample entries from training set:")
    with open("train.test.txt", 'r') as f:
        for _ in range(3):
            line = f.readline().strip()
            encoded_problem, encoded_solution = line.split('\t')
            
            # Decode for display
            problem = base64.b64decode(encoded_problem).decode('utf-8')
            solution = base64.b64decode(encoded_solution).decode('utf-8')
            
            print(f"Encoded: {encoded_problem}\t{encoded_solution}")
            print(f"Decoded: {problem}\t{solution}")
            print()

if __name__ == "__main__":
    main()
