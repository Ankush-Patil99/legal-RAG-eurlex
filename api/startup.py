import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration

class AppState:
    embedding_model = None
    generator = None
    tokenizer = None
    index = None
    metadata = None

def load_resources():
    AppState.embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    AppState.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    AppState.generator = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base"
    )
    AppState.generator.eval()

    AppState.index = faiss.read_index("indexes/faiss/eurlex_faiss.index")

    with open("data/embeddings/eurlex_metadata.json", "r", encoding="utf-8") as f:
        AppState.metadata = json.load(f)
