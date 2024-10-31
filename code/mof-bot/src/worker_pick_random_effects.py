# worker_pick_random_effects.py

import numpy as np
from enum import Enum

class Effect:
    def __init__(self, emojis, emotion, tone, length):
        """
        Initializes an Effect with the specified properties.
        
        Args:
            emojis (bool): Whether emojis are included.
            emotion (str): The selected emotion from a predefined set.
            tone (str): The selected tone from a predefined set.
            length (int): The "length" of the effect, based on a normal distribution.
        """
        self.emojis = emojis
        self.emotion = emotion
        self.tone = tone
        self.length = length

    def __repr__(self):
        return (f"Effect(emojis={self.emojis}, emotion='{self.emotion}', "
                f"tone='{self.tone}', length={self.length})")

def pick_effects():
    """
    Generates random effects for an object, with each property chosen randomly.
    
    Returns:
        Effect: An Effect object with randomized properties.
    """
    # Define possible values for each property
    emotions = ["angry", "embarrassed", "happy", "orgasmic", "exhausted", "bored", "amused"]
    tones = ["childish", "mature", "cryptobro", "fangirl", "robot"]

    # Generate each effect attribute randomly
    emojis = np.random.choice([True, False])
    emotion = np.random.choice(emotions)
    tone = np.random.choice(tones)
    length = int(np.clip(np.random.normal(80, 40), 10, 200))  # Normal dist with mean 80, clipped between 10 and 200

    # Create and return an Effect object
    return Effect(emojis=emojis, emotion=emotion, tone=tone, length=length)
