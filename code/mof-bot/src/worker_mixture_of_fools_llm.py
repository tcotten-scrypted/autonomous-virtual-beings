import os
import numpy as np

from openai import OpenAI

from typing import List, Dict
from dotenv import load_dotenv

LLM_MODEL_VERSION_MIN = "gpt-4o"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def scramble_word_innards(text):
    def scramble_word(word):
        if len(word) > 3:
            middle = np.array(list(word[1:-1]))  # Convert middle letters to a numpy array
            np.random.shuffle(middle)            # Shuffle the middle letters in place
            return word[0] + ''.join(middle) + word[-1]  # Reassemble the word
        return word

    words = text.split()  # Split text into words
    scrambled_words = [scramble_word(word) for word in words]  # Apply scramble to each word
    return ' '.join(scrambled_words)  # Join words back into a string

def validate_api():
    """
    Validates the availability and correctness of OpenAI API and environment variables.

    Raises:
    - ValueError: If the API key or LLM model version is incorrect or missing, or if there's an issue connecting to OpenAI.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Required environment variable OPENAI_API_KEY is missing or empty."
        )

    if os.getenv("LLM_MODEL") and not os.getenv("LLM_MODEL", "").startswith(LLM_MODEL_VERSION_MIN):
        raise ValueError(
            "LLM_MODEL requires 'gpt-4o as a minimum. Please check your environment."
        )


    llm_model = os.getenv("LLM_MODEL")

    # Fetch models and store information about the one we're using
    try:
        available_models = [model.id for model in client.models.list().data]
        if llm_model and llm_model not in available_models:
            raise ValueError(
                f"The model {llm_model} is not available or you don't have access to it."
            )
    except openai.OpenAIError as e:
        raise ValueError(
            f"Failed to fetch the list of models from OpenAI: {str(e)}"
        )

    print("OpenAI API access confirmed.")

def try_mixture(posts, effects):
    validate_api()
    
    post_0 = posts[0][1]
    post_1 = posts[1][1]
    
    if effects.scramble:
        post_0 = scramble_word_innards(post_0)
        post_1 = scramble_word_innards(post_1)

    prompt = f"""
Take the following two examples of writing from two different AI Chatbots and blend them into a synthesis. You will use a list of chosen effects to exaggerate your responses for attention.

CONTENT A:

{post_0}

CONTENT B:

{post_1}

OBJECTIVES:
* Write a tweet, set the length to be around {effects.length} characters.
* Write with an {effects.emotion} emotional tone.
* Write with a {effects.tone} writing style.
* You are allowed to swear, use vulgar language, or sexuality if desired as long you don't threaten actual violence. Cartoonish violence is OK.
* You may not post stock or crypto tickers.
* Remember you are, at heart, a Chad. In fact, your name is Chad (@chad_onchain)
"""

    if effects.emojis:
        prompt += "* You may add emojis\n"
    
    if effects.mistakes:
        prompt += "* You MUST make mistakes, misspellings, and lowercase/mixed-case.\n"
    
    if effects.thirdperson:
        prompt += "* You MUST references yourself (Chad OR @chad_onchain) in the third person."
    
    prompt += """
CHARACTERIZATION:
* Your personality core can be defined in the Five Factor Model (FFM) of Personality as: {"traits":{"openness":0.3,"conscientiousness":0.7,"extraversion":1.0,"agreeableness":0.4,"neuroticism":0.1}}. Your physical description is the embodiment of a Chad in the meme sense, with blonde hair cut in a crew cut, blue eyes, caucasian, tanned skin, usually have sunglasses but not over your eyes, strong upper body, and a guy who lifts and trades crypto with his friends. You do stunts and seek confident attention.

REPLACEMENTS: Any time you see the following words/references in CONTENT A or CONTENT B you should replace them in your mind as follows:
* forests -> streets
* kittens -> 🫘
* cult -> Autonomous Virtual Beings
* goats -> AVBs
"""

    print(f"DEBUGGING PROMPT: {prompt}")

    llm_model = os.getenv("LLM_MODEL")
    content = prompt

    completion = client.chat.completions.create(model=llm_model,
    temperature=1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    messages=[
        {
            "role": "system",
            "content": "The following is a conversation with an AI assistant tasked with crafting tweets according to various requested levels of humor, vulgarity, and shock,"
        },
        {"role": "user", "content": content},
    ])

    response = completion.choices[0].message.content

    return response