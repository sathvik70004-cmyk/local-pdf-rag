import os

DATA_DIR = "data"                 # where your files live
DB_PATH = os.environ.get("RAG_DB_PATH", "chroma_db")           # folder where ChromaDB saves vectors
TIMINGS_CSV = os.environ.get("RAG_TIMINGS_CSV", "timings.csv")  # one row per ingest run
COLLECTION_NAME = "docs"          # name of our collection (like a table)
EMBED_MODEL = "embeddinggemma"    # Gemma model that makes vectors
EMBED_DIM = None                  # None = full size (768). embeddinggemma also allows 512, 256, 128
LLM_MODEL = "gemma3:4b"           # Gemma model that writes answers
LLM_NUM_CTX = 8192                # how much text Gemma can read at once (tokens); raise for big chunks
TOP_K = 3                         # how many chunks to retrieve per question

# ---- chunking ----
CHUNKER = "paragraph"             # default strategy: paragraph | recursive | character | semantic
CHUNK_SIZE = 350                  # max characters per chunk (recursive, character)
CHUNK_OVERLAP = 50                # characters shared by neighbouring chunks (recursive, character)
MIN_CHARS = 150                   # paragraph strategy: merge paragraphs shorter than this
MAX_CHARS = 350                   # paragraph strategy: split paragraphs longer than this
