import re
from langchain_chroma import Chroma
from config import DB_PATH, COLLECTION_NAME, EMBED_MODEL, EMBED_DIM, TOP_K
from models import embeddings

# One collection per (model, dimension): vectors of different sizes can never be mixed.
# e.g. "docs_embeddinggemma_256" or "docs_embeddinggemma_full"
collection_name = re.sub(r"[^A-Za-z0-9._-]", "-", f"{COLLECTION_NAME}_{EMBED_MODEL}_{EMBED_DIM or 'full'}")

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,                      # Chroma calls this to embed text
    persist_directory=DB_PATH,                          # saved on disk
    collection_metadata={"hnsw:space": "cosine"},       # compare vectors by cosine distance
)


def add_chunks(chunks):
    ids = list(chunks)                                  # chunks is {id: Document}
    vectorstore.add_documents([chunks[i] for i in ids], ids=ids)   # embeds + stores (the slow step)


def get_ids(source):
    # All chunk IDs stored for one file. include=[] means "IDs only, skip the data".
    return vectorstore.get(where={"source": source}, include=[])["ids"]


def delete_ids(ids):
    if ids:                                             # Chroma errors on an empty list
        vectorstore.delete(ids=ids)


def count():
    return len(vectorstore.get(include=[])["ids"])


def search(question, k=TOP_K):
    # returns [(Document, distance), ...]; lower distance = more similar
    return vectorstore.similarity_search_with_score(question, k=k)