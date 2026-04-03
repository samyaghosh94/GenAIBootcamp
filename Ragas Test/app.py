import os
import openai
from ragas import evaluate
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ragas.metrics.collections import harmfulness
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, SemanticSimilarity, AnswerCorrectness
from ragas import Dataset