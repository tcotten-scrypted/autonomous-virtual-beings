import json
import os

data_file_path = os.path.join(os.path.dirname(__file__), "../data/posts.json")

available_content = None
num_fools = 0
num_posts_per_fool = []

def summarize():
    global num_fools, num_posts_per_fool
    
    if available_content is None:
        print("No content loaded.")
        return
    
    # Count the number of keys in available_content
    num_fools = len(available_content)
    
    # Generate an array of post counts for each key in available_content
    num_posts_per_fool = [len(posts) for posts in available_content.values()]
    
    print(f"Number of fools: {num_fools}")
    print(f"Number of posts per fool: {num_posts_per_fool}")

def load_available_content():
    """
    Loads JSON data from posts.json and stores it in available_content.
    Expects posts.json to contain an object (dictionary).
    """
    global available_content
    try:
        with open(data_file_path, "r") as file:
            available_content = json.load(file)
            summarize()
    except FileNotFoundError:
        print(f"Error: {data_file_path} not found.")
        available_content = {}  # Set as an empty dictionary if the file is missing
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        available_content = {}  # Set as an empty dictionary if decoding fails