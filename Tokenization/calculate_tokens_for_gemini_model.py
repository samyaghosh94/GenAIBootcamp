import os

import google.generativeai as genai

genai.configure(api_key="AIzaSyABtGvfrK1i3K588x4FIYRzzWRR9z7hOjg")
image_path = r'C:\Users\samya_ghosh\PycharmProjects\GenAIBootcamp\images\amazon_snapshot.png'
assert os.path.exists(image_path), f"Test image not found!: {image_path}"

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(["""
    Can you please tell me how to login?
    1. Please explain in **simple steps** how a user can log in, using natural language.

    2. Additionally, extract **only the UI elements relevant to login**, and return them as a JSON array with:
        - type (e.g., 'text', 'icon', 'button')
        - content (e.g., 'Hello, sign in')
        - bounding box (normalized [x1, y1, x2, y2])
        - interactivity (true/false)

        Return the explanation first, then the JSON block.
    """,
                                   {"mime_type": "image/png", "data": open(image_path, "rb").read()}])

# Print the actual output
print("Response:", response.text)

# Access token usage info
print("Prompt tokens:", response.usage_metadata.prompt_token_count)
print("Response tokens:", response.usage_metadata.candidates_token_count)
print("Total tokens:", response.usage_metadata.total_token_count)
