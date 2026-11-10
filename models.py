import math
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
import sys
from config import EMBED_MODEL, EMBED_DIM, LLM_MODEL, LLM_NUM_CTX

MRL_DIMS = (768, 512, 256, 128)        # sizes embeddinggemma was trained to support (Matryoshka)


class TruncatedEmbeddings(Embeddings):
    # Wraps another embedding model and keeps only the first `dim` numbers of each vector.
    # embeddinggemma is trained so that those first numbers still carry most of the meaning.
    def __init__(self, base, dim):
        self.base = base
        self.dim = dim

    def _fix(self, vec):
        v = vec[: self.dim]                            # keep the first `dim` numbers
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]                   # rescale to length 1 (needed for cosine)

    def embed_documents(self, texts):
        return [self._fix(v) for v in self.base.embed_documents(texts)]

    def embed_query(self, text):
        return self._fix(self.base.embed_query(text))


base_embeddings = OllamaEmbeddings(model=EMBED_MODEL)      # full-size vectors
if EMBED_DIM is None:
    embeddings = base_embeddings                           # no truncation
else:
    if not EMBED_MODEL.startswith("embeddinggemma"):
        raise ValueError("--dim only works with embeddinggemma (it is the Matryoshka model)")
    if not 1 <= EMBED_DIM <= 768:
        raise ValueError("dimension must be between 1 and 768")
    if EMBED_DIM not in MRL_DIMS:
        print(f"Note: {EMBED_DIM} is not a size the model was trained for ({MRL_DIMS}); "
              "quality may drop sharply.", file=sys.stderr)
    embeddings = TruncatedEmbeddings(base_embeddings, EMBED_DIM)

llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=LLM_NUM_CTX)   # writes answers; num_ctx = reading limit