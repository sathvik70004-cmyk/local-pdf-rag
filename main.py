import csv
import sys
import time
import argparse
from pathlib import Path
from loaders import load_file, LOADERS
import config
from chunker import chunk_documents, STRATEGIES
from config import CHUNKER


def log_row(row):
    path = Path(config.TIMINGS_CSV)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["label", "chunker", "mode", "file", "total_chunks", "embedded", "deleted",
                        "skipped", "load_s", "chunk_s", "embed_store_s", "total_s"])
        w.writerow(row)


def expand(paths):
    # Turn a mix of files and folders into a flat list of supported files.
    files = []
    for p in map(Path, paths):
        if p.is_dir():                                    # a folder: take every file we have a loader for
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in LOADERS)
        else:
            files.append(p)
    return files


def ingest(path, full, label, chunker, size, overlap):
    from store import add_chunks, get_ids, delete_ids, count      # imported here: preview needs no DB
    from models import embeddings
    name = Path(path).name
    embeddings.embed_query("warm-up")                 # loads the model BEFORE timing starts

    t0 = time.perf_counter()
    docs = load_file(path)                            # stage 1: PDF -> LangChain Documents
    t1 = time.perf_counter()
    chunks = chunk_documents(docs, chunker, size, overlap)   # stage 2: strategy + hash IDs
    t2 = time.perf_counter()

    if not chunks:
        print("No text found (is the PDF scanned images?).")
        return

    longest = max(len(c.page_content) for c in chunks.values())
    if longest > 6000:                                # embeddinggemma reads ~2000 tokens, then silently cuts
        print(f"WARNING: longest chunk is {longest} characters; text past ~6000 characters is ignored by the embedding model.")

    old_ids = set(get_ids(name))                      # what is stored now
    new_ids = set(chunks)                             # what the file says now

    if full:                                          # wipe and rebuild
        to_delete = list(old_ids)
        to_add = chunks
    else:                                             # the diff
        to_delete = list(old_ids - new_ids)
        to_add = {i: c for i, c in chunks.items() if i not in old_ids}

    delete_ids(to_delete)
    if to_add:
        add_chunks(to_add)                            # embedding happens here
    t3 = time.perf_counter()

    mode = "full" if full else "incremental"
    used = chunker or CHUNKER
    skipped = len(chunks) - len(to_add)
    load_s, chunk_s, es_s, total_s = t1 - t0, t2 - t1, t3 - t2, t3 - t0

    print(f"\nMode: {mode} | chunker: {used} | embedding dimension: {config.EMBED_DIM or 768}")
    print(f"Chunks in file: {len(chunks)} | embedded: {len(to_add)} | "
          f"deleted: {len(to_delete)} | skipped: {skipped}")
    print(f"Load {load_s:.3f}s | Chunk {chunk_s:.3f}s | "
          f"Embed+store {es_s:.3f}s | TOTAL {total_s:.3f}s")
    print(f"DB total: {count()}")

    log_row([label, used, mode, name, len(chunks), len(to_add), len(to_delete), skipped,
             f"{load_s:.4f}", f"{chunk_s:.4f}", f"{es_s:.4f}", f"{total_s:.4f}"])


def preview(path, chunker, size, overlap, show):
    # Chunk the file and print statistics. Touches NO database and does NO embedding
    # (except the semantic strategy, which needs embeddings to find topic changes).
    docs = load_file(path)
    names = list(STRATEGIES) if chunker == "all" else [chunker]
    print(f"{len(docs)} pages | {sum(len(d.page_content) for d in docs)} characters in total\n")
    print(f"{'chunker':<11}{'chunks':>7}{'min':>6}{'avg':>6}{'max':>6}")
    for n in names:
        try:
            chunks = chunk_documents(docs, n, size, overlap)
        except ImportError:                           # semantic needs langchain-experimental
            print(f"{n:<11} skipped (pip install langchain-experimental)")
            continue
        lens = [len(c.page_content) for c in chunks.values()]
        print(f"{n:<11}{len(lens):>7}{min(lens):>6}{sum(lens) // len(lens):>6}{max(lens):>6}")
        if show:
            for i, c in enumerate(chunks.values(), start=1):
                print(f"  [{i}] (page {c.metadata['page']}, {len(c.page_content)} chars) {c.page_content}")
            print()


def show_answer(q):
    from rag import ask
    from store import count
    if count() == 0:
        print(f"\nNothing stored for dimension {config.EMBED_DIM or 768}. "
              f"Ingest first, with the same --dim.")
        return
    answer, docs, dists = ask(q)
    print(f"\n(embedding dimension: {config.EMBED_DIM or 768})")

    print("\n===== RETRIEVED CHUNKS =====")
    for i, (d, dist) in enumerate(zip(docs, dists), start=1):
        print(f"\n[{i}] distance={dist:.4f} | {d.metadata['source']} | page {d.metadata.get('page', '?')}")
        print(d.page_content)

    print("\n===== COMBINED ANSWER =====")
    print(answer)


def chat(question=None):
    if question:                                      # one-shot mode: answer once and exit
        show_answer(question)
        return
    while True:                                       # interactive mode
        q = input("\nAsk a question (or type exit): ")
        if q.strip().lower() == "exit":
            break
        if not q.strip():
            continue
        show_answer(q)


def build_parser():
    p = argparse.ArgumentParser(description="Local PDF RAG (LangChain + Ollama + ChromaDB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def chunk_opts(sp, choices):
        sp.add_argument("--chunker", choices=choices, help=f"chunking strategy (default: {CHUNKER})")
        sp.add_argument("--size", type=int, help="max chunk size in characters")
        sp.add_argument("--overlap", type=int, help="overlap in characters")

    ing = sub.add_parser("ingest", help="load a file into the database")
    ing.add_argument("files", nargs="+", help="one or more files, or a folder")
    ing.add_argument("--full", action="store_true", help="wipe this file's chunks and rebuild")
    ing.add_argument("--label", default="", help="name for this run in timings.csv")
    ing.add_argument("--dim", type=int, help="embedding dimension: 768, 512, 256, 128 (default: full)")
    chunk_opts(ing, list(STRATEGIES))

    pre = sub.add_parser("chunks", help="preview chunking without touching the database")
    pre.add_argument("file")
    pre.add_argument("--show", action="store_true", help="print every chunk")
    chunk_opts(pre, list(STRATEGIES) + ["all"])

    ask_p = sub.add_parser("ask", help="ask questions about the ingested files")
    ask_p.add_argument("question", nargs="*", help="optional: ask once and exit")
    ask_p.add_argument("--dim", type=int, help="embedding dimension to search (must match the ingest)")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles choke on PDF symbols
    a = build_parser().parse_args()
    if getattr(a, "dim", None):
        config.EMBED_DIM = a.dim                      # must happen BEFORE models.py / store.py are imported
    if a.cmd == "ingest":
        files = expand(a.files)
        if not files:
            print("No supported files found.")
        for f in files:                                   # each file is diffed against its own stored chunks
            print(f"\n--- {f.name} ---")
            ingest(str(f), a.full, a.label, a.chunker, a.size, a.overlap)
    elif a.cmd == "chunks":
        preview(a.file, a.chunker or CHUNKER, a.size, a.overlap, a.show)
    elif a.cmd == "ask":
        chat(" ".join(a.question) or None)       # no question given -> interactive loop    