# worker_pick_random_effects.py

import numpy as np
from enum import Enum

class Effect:
    def __init__(self, usethird, tag, emojis, emotion, tone, length, lore, scramble, mistakes, thirdperson, sickjoke, freeukraine):
        """
        Initializes an Effect with the specified properties.
        
        Args:
            emojis (bool): Whether emojis are included.
            emotion (str): The selected emotion from a predefined set.
            tone (str): The selected tone from a predefined set.
            length (int): The "length" of the effect, based on a normal distribution.
        """
        self.usethird = usethird
        self.tag = tag
        self.emojis = emojis
        self.emotion = emotion
        self.tone = tone
        self.length = length
        self.lore = lore
        self.scramble = scramble
        self.mistakes = mistakes
        self.thirdperson = thirdperson
        self.sickjoke = sickjoke
        self.freeukraine = freeukraine

    def __repr__(self):
        return (f"Effect(usethird={self.usethird}, tag={self.tag}, emojis={self.emojis}, emotion='{self.emotion}', "
                f"tone='{self.tone}', length={self.length}, lore={self.lore}, "
                f"scramble={self.scramble}, mistakes={self.mistakes}, thirdperson={self.thirdperson}, "
                f"sickjoke={self.sickjoke}, self.freeukraine={self.freeukraine}")

def pick_effects():
    """
    Generates random effects for an object, with each property chosen randomly.
    
    Returns:
        Effect: An Effect object with randomized properties.
    """
    # Define possible values for each property
    emotions = ["confident", "triumphant", "enthusiastic", "prideful", "satisfied", "determined", "thrilled", "amused", "dominating", "curious", "excited", "competitive", "bold", "joyful", "motivated", "victorious", "calm", "seld-assured", "orgasmic", "exhausted", "bored", "frustrated"]
    tones = ["alpha", "motivational guru", "american psycho", "investment banker", "vitalik buterin", "cryptobro", "cryptogirl", "influencer", "beastmode", "minimalist"]

    # Generate each effect attribute randomly
    usethird = np.random.choice([True, False])
    tag = np.random.choice([True, False], p=[0.33, 0.67])
    emojis = np.random.choice([True, False], p=[0.05, 0.95])
    emotion = np.random.choice(emotions)
    tone = np.random.choice(tones)
    length = int(np.clip(np.random.normal(40, 40), 10, 220))
    lore = np.random.choice([True, False], p=[0.05, 0.95])
    scramble = np.random.choice([True, False], p=[0.1, 0.9])
    mistakes = np.random.choice([True, False], p=[2/3, 1/3])
    thirdperson = np.random.choice([True, False], p=[0.01, 0.99])
    sickjoke = np.random.choice([True, False], p=[0.01, 0.99])
    freeukraine = np.random.choice([True, False], p=[0.001, 0.999])
    
    if lore:
        length *= 4

    # Create and return an Effect object
    return Effect(usethird=usethird, tag=tag, emojis=emojis, emotion=emotion, tone=tone, length=length, lore=lore, scramble=scramble, mistakes=mistakes, thirdperson=thirdperson, sickjoke=sickjoke, freeukraine=freeukraine)
