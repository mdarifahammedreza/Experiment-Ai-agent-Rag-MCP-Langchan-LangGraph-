import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message["content"])