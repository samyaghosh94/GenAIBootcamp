import requests
import os

from dotenv import load_dotenv

load_dotenv()

# Set your API key and endpoint URL
API_KEY = os.getenv("DIAL_LAB_KEY")
ENDPOINT = "https://ai-proxy.lab.epam.com/openai/deployments/text-embedding-3-small-1/embeddings"

# Define a simple test function to verify the connection and API functionality
def test_embedding_connection():
    # Define the text you want to embed
    text_to_embed = "This is a test to check the embedding connection."

    # Prepare the payload for the API request
    payload = {
        "input": [text_to_embed]
    }

    # Prepare the headers with the API key
    headers = {
        "Api-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Make the POST request to the embedding API
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Successfully connected to the embedding API!")
            print("Response:", response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        print("Error: The request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Run the test function
if __name__ == "__main__":
    test_embedding_connection()
