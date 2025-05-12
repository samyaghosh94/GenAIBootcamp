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
                return data
    except Exception as e:
        print(f"❌ Exception while calling Gemini API: {e}")
        return None
