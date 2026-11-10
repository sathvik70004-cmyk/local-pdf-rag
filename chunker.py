import re
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from config import CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS, MAX_CHARS


def make_id(source, text):
    # The ID is a fingerprint of the content: same text -> same ID, always.
    raw = f"{source}\n{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------- Strategy 1: paragraph (our own splitter, wrapped for LangChain) ----------
def _split_long(par):
    sentences = re.split(r"(?<=[.!?])\s+", par)         # split after . ! ? followed by space
    pieces, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > MAX_CHARS:   # adding s would overflow
            pieces.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _split_paragraphs(text):
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    out, buf = [], ""
    for p in paras:
        buf = f"{buf} {p}".strip()
        if len(buf) >= MIN_CHARS:                       # big enough to stand alone
            out.extend(_split_long(buf) if len(buf) > MAX_CHARS else [buf])
            buf = ""
    if buf:
        out.append(buf)
    return out


def paragraph_split(docs, size, overlap):
    # size/overlap are ignored: this strategy uses MIN_CHARS / MAX_CHARS from config.py
    out = []
    for d in docs:
        for piece in _split_paragraphs(d.page_content):
            out.append(Document(page_content=piece, metadata=dict(d.metadata)))
    return out


# ---------- Strategy 2: recursive (LangChain's recommended default) ----------
def recursive_split(docs, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)               # tries \n\n, then \n, then space, then characters


# ---------- Strategy 3: character (plain fixed-size slices) ----------
def character_split(docs, size, overlap):
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)               # cuts every `size` characters, ignores meaning


# ---------- Strategy 4: semantic (splits where the meaning changes) ----------
def semantic_split(docs, size, overlap):
    from langchain_experimental.text_splitter import SemanticChunker   # pip install langchain-experimental
    from models import embeddings
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return splitter.split_documents(docs)               # embeds every sentence, so it is the slowest


STRATEGIES = {
    "paragraph": paragraph_split,
    "recursive": recursive_split,
    "character": character_split,
    "semantic": semantic_split,
}


def chunk_documents(docs, strategy=None, size=None, overlap=None):
    strategy = strategy or CHUNKER                      # fall back to config.py
    size = size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunker '{strategy}'. Choose from: {', '.join(STRATEGIES)}")

    chunks = {}                                         # dict keyed by ID also removes duplicates
    for piece in STRATEGIES[strategy](docs, size, overlap):
        text = piece.page_content.strip()
        if not text:
            continue
        cid = make_id(piece.metadata["source"], text)
        chunks[cid] = Document(page_content=text, metadata=piece.metadata)
    return chunks                                       # {id: Document}