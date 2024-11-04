import os
import json
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/lore.json")

def load_lore_data(filepath=DATA_FILE):
    """
    Loads lore data from a specified JSON file.
    Returns the data as a dictionary.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        lore_data = json.load(file)
    return lore_data

def pick_lore():
    """
    Selects a random topic and its content from the lore data.
    Returns a dictionary with the selected topic and its content.
    """
    
    lore_data = load_lore_data()
    
    # Ensure there is content to choose from
    if not lore_data or len(lore_data) < 1:
        raise ValueError("Insufficient data: The lore data is empty or missing")

    # Randomly pick a topic
    topic = str(np.random.choice(list(lore_data.keys())))

    # Create an object with the topic and its content
    selected_lore = {
        "topic": topic,
        "content": lore_data[topic]
    }

    return selected_lore

# Example usage
if __name__ == "__main__":
    selected_lore = pick_lore(lore_data)
    print("Selected Lore Entry:")
    print(selected_lore)
