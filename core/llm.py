import os
from typing import List, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize Groq client with a timeout to prevent hanging threads
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30.0,
)

# Recommended model for fast, high-quality reasoning
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def generate_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
) -> str:
    """Sends a chat completion request to Groq."""
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"Error calling Groq API: {exc}")
        return "Sorry, I hit a temporary AI service issue. Please try again in a moment."
