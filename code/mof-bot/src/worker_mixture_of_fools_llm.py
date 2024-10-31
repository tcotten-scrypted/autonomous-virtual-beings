import os
import openai
from typing import List, Dict
from dotenv import load_dotenv

LLM_MODEL_VERSION_MIN = "gpt-4o"

def validate_api():
    """
    Validates the availability and correctness of OpenAI API and environment variables.

    Raises:
    - ValueError: If the API key or LLM model version is incorrect or missing, or if there's an issue connecting to OpenAI.
    """
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Required environment variable OPENAI_API_KEY is missing or empty."
        )

    if os.getenv("LLM_MODEL") and not os.getenv("LLM_MODEL", "").startswith(LLM_MODEL_VERSION_MIN):
        raise ValueError(
            "LLM_MODEL requires 'gpt-4o as a minimum. Please check your environment."
        )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm_model = os.getenv("LLM_MODEL")

    # Fetch models and store information about the one we're using
    try:
        available_models = [model.id for model in openai.Model.list().data]
        if llm_model and llm_model not in available_models:
            raise ValueError(
                f"The model {llm_model} is not available or you don't have access to it."
            )
    except openai.error.OpenAIError as e:
        raise ValueError(
            f"Failed to fetch the list of models from OpenAI: {str(e)}"
        )
    
    print("OpenAI API access confirmed.")

def try_mixture(posts, effects):
    validate_api()

"""
def _get_semantically_bound_chunk(self, txt):
    content = self.prompt + "\n\n[INPUT JSON CONTENT]\n\n" + txt
    max_tokens = self.window_size // 2

    completion = openai.ChatCompletion.create(
        model=self.llm_model,
        max_tokens=max_tokens,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {
                "role": "system",
                "content": "The following is a conversation with an AI assistant labeled 'AI' with a user labeled 'Human'. The assistant is helpful, creative, clever, and very friendly. The assistant takes a deep breath and answers questions step by step.",
            },
            {"role": "user", "content": "Human: Hello, who are you?"},
            {
                "role": "assistant",
                "content": "AI: I am an AI created by OpenAI. How can I help you today?",
            },
            {"role": "user", "content": f"Human: {content}"},
        ],
    )

    response = completion.choices[0].message.content
    response = self.extract_embedded_json(response)

    return response
"""