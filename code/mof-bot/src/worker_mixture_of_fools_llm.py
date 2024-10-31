import os
from openai import OpenAI

from typing import List, Dict
from dotenv import load_dotenv

LLM_MODEL_VERSION_MIN = "gpt-4o"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    prompt = f"""
Take the following two examples of writing from two different AI Chatbots and blend them into a synthesis. You will use a list of chosen effects to exaggerate your responses with a certain emotion, tone, and length.

CONTENT A:

{posts[0][1]}

CONTENT B:

{posts[1][1]}

OBJECTIVES:
* Write a tweet, choose the length to be around {effects.length} characters.
* Choose an emotional tone of: {effects.emotion}.
* Choose a writing style of: {effects.tone}.
* You are allowed to swear, use vulgar language, or sexuality if desired.
* You may not post stock or crypto tickers.
"""

    if effects.emojis:
        prompt += "\n* You may add emojis"

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