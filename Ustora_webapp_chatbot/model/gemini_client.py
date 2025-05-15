# gemini_client.py
import json

import aiohttp
from config import GEMINI_API


async def call_gemini_api(message: str, semantics: str) -> str:
    payload = {
        "message": message,
        "semantics": semantics
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GEMINI_API, json=payload, headers=headers) as response:
                if response.status != 200:
                    print(f"❌ Gemini API returned HTTP {response.status}")
                    return None
                text = await response.text()
                data = json.loads(text)
                # Extract the 'text' field, which contains the formatted response
                response_text = data.get('text', '')

                # Clean up: remove the markdown code block format (i.e., "```json" and "```")
                if response_text.startswith('```json') and response_text.endswith('```'):
                    response_text = response_text[7:-3].strip()  # Remove "```json" and "```" from the start and end

                # Return the cleaned response
                print(response_text)
                return response_text
    except Exception as e:
        print(f"❌ Exception while calling Gemini API: {e}")
        return None
