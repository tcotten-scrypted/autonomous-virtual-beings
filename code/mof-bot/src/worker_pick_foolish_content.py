import numpy as np

def pick_two_posts(fools_content):
    """
    Selects one random post from two distinct random fools in available_content.
    Returns a list of tuples with the selected fools and their posts.
    """
    # Ensure there is enough content to select two different fools
    if fools_content.available_content is None or len(fools_content.available_content) < 2:
        raise ValueError("Insufficient data: At least two fools are required")

    # Randomly pick two unique fools
    fools = list(fools_content.available_content.keys())
    selected_fools = np.random.choice(fools, size=2, replace=False)

    # Pick one random post from each selected fool
    selected_posts = []
    for fool in selected_fools:
        post = np.random.choice(fools_content.available_content[fool])
        selected_posts.append((fool, post))

    return selected_posts