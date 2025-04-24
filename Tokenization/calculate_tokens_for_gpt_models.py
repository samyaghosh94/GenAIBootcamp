import tiktoken

def count_tokens(text):
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    token_data = {}

    for model in models:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback if model is not recognized
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(text)
        decoded_tokens = [enc.decode([t]) for t in tokens]

        token_data[model] = {
            "count": len(tokens),
            "tokens": tokens,
            "decoded_tokens": decoded_tokens,
        }

    return token_data


# Example usage
text = "This is an example text to calculate token usage."
token_data = count_tokens(text)

for model, data in token_data.items():
    print(f"\nModel: {model}")
    print(f"Token count: {data['count']}")
    print("Tokens (IDs):", data['tokens'])
    print("Decoded tokens:", data['decoded_tokens'])
