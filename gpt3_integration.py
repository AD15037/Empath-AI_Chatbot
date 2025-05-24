import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key in environment variables

def generate_response(user_message):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_message,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()