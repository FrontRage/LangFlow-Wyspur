from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create an instance of the OpenAI Class
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text_basic(
    prompt: str, 
    model: str, 
    temperature: float = 0.0, 
    top_p: float = 1.0
):
    """
    Generates text using the chat.completions.create endpoint, 
    with optional temperature and top_p parameters.
    
    :param prompt: The user prompt.
    :param model: The model ID (e.g. "gpt-4o" or "gpt-4o-mini").
    :param temperature: Sampling temperature (0..2). Higher = more creative, lower = more strict.
    :param top_p: Nucleus sampling (0..1). 
    :return: The assistant's message content.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content



def generate_text_with_conversation(messages,model = "gpt-3.5-turbo"):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
        )
    return response.choices[0].message.content