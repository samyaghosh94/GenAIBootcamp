import requests as r
import os
import json
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("DIAL_LAB_KEY")
models = r.get("https://ai-proxy.lab.epam.com/openai/models",
               headers={"Api-Key": api_key}).json()["data"]

# Filter models with 'text-embedding' in their id
embedding_models = [model for model in models if "text-embedding" in model["id"]]
print(embedding_models)

# Get model limits
result = {}
for model in models:
    model_id = model["id"]
    try:
        limits = r.get(f"https://ai-proxy.lab.epam.com/v1/deployments/{model_id}/limits",
                              headers={"Api-Key": api_key}).json()
        minute_limit = limits.get("minuteTokenStats", {}).get("total", 0)
        day_limit = limits.get("dayTokenStats", {}).get("total", 0)
        if minute_limit > 0 or day_limit > 0:
            result[model_id] = {"limits": {"minute": minute_limit, "day": day_limit}}
    except:
        pass

print(result)

# Get model data for a specific model
model = "gpt-4o-mini-2024-07-18"
model_data = r.get(f"https://ai-proxy.lab.epam.com/openai/models/{model}",
               headers={"Api-Key": api_key}).json()
print(model_data)