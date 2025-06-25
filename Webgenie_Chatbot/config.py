# config.py

# Webapp URL to capture screenshots from
WEBAPP_URL = 'http://localhost:4200'  # Change this to your webapp URL

# OmniParser API endpoint
OMNIPARSER_API = 'http://localhost:8000/process/'

# Gemini API endpoint
GEMINI_API = 'http://localhost:5000/api/gemini/generate'

# Directory where screenshots are saved
SCREENSHOT_DIR = './screenshots'

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding model configuration
EMBEDDING_MODEL = 'openai'  # change to 'huggingface' or other if needed

FAISS_INDEX_PATH = "storage/faiss_index.index"
DOCSTORE_PATH = "storage/documents.json"
QNA_DOC_PATH = "storage/QnA Document for Ustora Website.txt"
QNA_PARSED_PATH = "storage/qna_data.json"
QNA_PARSED_TEXT_PATH = "storage/qna_texts.json"
DOCX_SOURCE_PATH = "storage/HCM_BackOffice_User_Help_Guide_Enhanced.docx"
