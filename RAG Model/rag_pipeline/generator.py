from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GENAI_KEY"))

def ask_gemini(query, context_docs):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

            Context:
            {context_docs}

            Question:
            {query}
            """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text
