# gemini_client.py

import json
import aiohttp
from config import GEMINI_API
from langsmith import traceable

@traceable
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
                    return "⚠️ Error calling Gemini API"
                text = await response.text()
                data = json.loads(text)
                response_text = data.get('text', '')

                # Clean up response
                if response_text.startswith('```json') and response_text.endswith('```'):
                    response_text = response_text[7:-3].strip()

                return response_text
    except Exception as e:
        print(f"❌ Exception while calling Gemini API: {e}")
        return "⚠️ Error calling Gemini"
