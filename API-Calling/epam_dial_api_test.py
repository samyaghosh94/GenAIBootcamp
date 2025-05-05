import requests as r
import os
from dotenv import load_dotenv
load_dotenv()


class LLM:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.headers = {"Api-Key": api_key}
        self.endpoint = f"{base_url}/openai/deployments/{self.model}/chat/completions"


    def generate(self, prompt: str) -> str:
        # Implement the function to create completion for the prompt and return text
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }
        response = r.post(self.endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

llm = LLM(model="gpt-4o-mini-2024-07-18", api_key=os.getenv("DIAL_LAB_KEY"), base_url="https://ai-proxy.lab.epam.com")

if __name__ == '__main__':
    prompt = "What is the capital of France?"

    # Call the generate method
    response = llm.generate(prompt)
    print(response)