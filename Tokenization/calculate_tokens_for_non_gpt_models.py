from transformers import AutoTokenizer

def count_tokens_with_autotokenizer(text, model_names):
    token_data = {}

    for model in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([t]) for t in tokens]

            token_data[model] = {
                "count": len(tokens),
                "tokens": tokens,
                "decoded_tokens": decoded_tokens
            }

        except Exception as e:
            token_data[model] = {"error": str(e)}

    return token_data


# Example usage
text = "This is an example text to calculate token usage."
models = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

token_data = count_tokens_with_autotokenizer(text, models)

for model, data in token_data.items():
    print(f"\nModel: {model}")
    if "error" in data:
        print(f"Error: {data['error']}")
    else:
        print(f"Token count: {data['count']}")
        print("Tokens (IDs):", data['tokens'])
        print("Decoded tokens:", data['decoded_tokens'])
