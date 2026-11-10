"""Experiment B: how embedding dimension (Matryoshka truncation) affects retrieval and answers.

Corpus : data/doc_40.pdf, chunked with the project's default strategy (paragraph).
Eval   : 30 hand-written questions (experiments/qa_git.py) with known evidence passages.
Dims   : 768 512 256 128 (trained Matryoshka sizes) + 64 32 (NOT trained; stress test).

Each dimension gets its own Chroma collection built with the project's own
TruncatedEmbeddings class and the same cosine HNSW settings as store.py. The 768-d
vectors are computed once and cached, so every dimension sees identical base vectors
(truncation is deterministic, so this is exactly what `--dim N` produces).

Metrics: hit@1/3/5, MRR@10, mean top-1 distance, distance spread (top10 - top1),
top-3 overlap with the 768-d result, and (for 768/512/256/128) LLM answer accuracy
and length with gemma3:4b at TOP_K=3.

usage: python experiments/exp_dimensions.py [--no-llm]
output: results/dimensions/summary.csv, summary.md, per_question.json, *.png
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
WORK = ROOT / "experiments" / "_work" / "dim_db"
os.environ["RAG_DB_PATH"] = str(WORK)                  # never touch the user's chroma_db
OUT = ROOT / "results" / "dimensions"

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from loaders import load_file
from chunker import chunk_documents
from models import base_embeddings, TruncatedEmbeddings, llm
from qa_git import QA, is_hit, is_correct

DIMS = [768, 512, 256, 128, 64, 32]
LLM_DIMS = [768, 512, 256, 128]
K = 10
TOP_K = 3


class Cached(Embeddings):
    """Remembers every vector so each text is embedded by Ollama only once."""
    def __init__(self, base):
        self.base, self.cache = base, {}

    def embed_documents(self, texts):
        missing = [t for t in texts if t not in self.cache]
        if missing:
            for i in range(0, len(missing), 32):          # small batches: big ones crashed the Ollama runner
                part = missing[i:i + 32]
                for t, v in zip(part, self.base.embed_documents(part)):
                    self.cache[t] = v
        return [self.cache[t] for t in texts]

    def embed_query(self, text):
        key = "\0q" + text
        if key not in self.cache:
            self.cache[key] = self.base.embed_query(text)
        return self.cache[key]


PROMPT = ("Answer the question using only the context below. "
          "If the answer is not in the context, say you don't know.\n\n"
          "Context:\n{context}\n\nQuestion: {question}")          # identical to rag.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--doc", default="data/doc_40.pdf")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)

    chunks = chunk_documents(load_file(a.doc))           # default chunker from config.py
    ids = list(chunks)
    docs = [chunks[i] for i in ids]
    print(f"{len(docs)} chunks from {a.doc}")
    cached = Cached(base_embeddings)
    t = time.perf_counter()
    cached.embed_documents([d.page_content for d in docs])
    embed_s = time.perf_counter() - t
    for item in QA:
        cached.embed_query(item["q"])

    rows, per_q = [], {}
    ref_top3 = {}
    for dim in DIMS:
        store = Chroma(collection_name=f"dimexp_{dim}", embedding_function=TruncatedEmbeddings(cached, dim),
                       persist_directory=str(WORK), collection_metadata={"hnsw:space": "cosine"})
        store.add_documents(docs, ids=ids)
        hits1 = hits3 = hits5 = 0
        rr, top1, spread, overlap = [], [], [], []
        correct, lengths, qtimes = 0, [], []
        for qi, item in enumerate(QA):
            t = time.perf_counter()
            res = store.similarity_search_with_score(item["q"], k=K)
            qtimes.append(time.perf_counter() - t)
            flags = [is_hit(d.page_content, item) for d, _ in res]
            rank = next((i + 1 for i, f in enumerate(flags) if f), None)
            hits1 += any(flags[:1]); hits3 += any(flags[:3]); hits5 += any(flags[:5])
            rr.append(1 / rank if rank else 0.0)
            dists = [s for _, s in res]
            top1.append(dists[0]); spread.append(dists[-1] - dists[0])
            top3 = [d.page_content for d, _ in res[:3]]
            if dim == 768:
                ref_top3[qi] = top3
            overlap.append(len(set(top3) & set(ref_top3[qi])) / 3)
            entry = {"dim": dim, "rank_of_evidence": rank,
                     "retrieved": [{"page": d.metadata.get("page"), "distance": round(s, 4),
                                    "hit": f, "text": d.page_content[:160]}
                                   for (d, s), f in zip(res[:5], flags[:5])]}
            if dim in LLM_DIMS and not a.no_llm:
                ctx = "\n\n".join(f"[{d.metadata['source']}, page {d.metadata.get('page', '?')}] {d.page_content}"
                                  for d, _ in res[:TOP_K])
                ans = llm.invoke(PROMPT.format(context=ctx, question=item["q"])).content
                ok = is_correct(ans, item)
                correct += ok
                lengths.append(len(ans.split()))
                entry.update(answer=ans, answer_correct=ok)
            per_q.setdefault(item["q"], []).append(entry)
        n = len(QA)
        row = {"dim": dim, "trained_mrl_size": dim in (768, 512, 256, 128),
               "bytes_per_vector": dim * 4,
               "hit@1": f"{hits1 / n:.3f}", "hit@3": f"{hits3 / n:.3f}", "hit@5": f"{hits5 / n:.3f}",
               "mrr@10": f"{statistics.mean(rr):.3f}",
               "mean_top1_distance": f"{statistics.mean(top1):.4f}",
               "mean_spread_top10": f"{statistics.mean(spread):.4f}",
               "top3_overlap_vs_768": f"{statistics.mean(overlap):.3f}",
               "median_query_ms": f"{1000 * statistics.median(qtimes):.1f}",
               "llm_accuracy": f"{correct / n:.3f}" if lengths else "",
               "mean_answer_words": f"{statistics.mean(lengths):.1f}" if lengths else ""}
        rows.append(row)
        print(row)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "per_question.json").write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["| dim | trained size | bytes/vector | hit@1 | hit@3 | hit@5 | MRR@10 | top-3 overlap vs 768 | mean top-1 dist | spread (d10-d1) | LLM accuracy | answer words |",
          "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['dim']} | {'yes' if r['trained_mrl_size'] else 'no'} | {r['bytes_per_vector']} | {r['hit@1']} | "
                  f"{r['hit@3']} | {r['hit@5']} | {r['mrr@10']} | {r['top3_overlap_vs_768']} | {r['mean_top1_distance']} | "
                  f"{r['mean_spread_top10']} | {r['llm_accuracy'] or '-'} | {r['mean_answer_words'] or '-'} |")
    md.append(f"\nCorpus: {a.doc}, {len(docs)} paragraph chunks; {len(QA)} questions; "
              f"embedding all chunks once took {embed_s:.2f}s.")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [str(r["dim"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for key in ["hit@1", "hit@3", "mrr@10", "top3_overlap_vs_768"]:
        axes[0].plot(xs, [float(r[key]) for r in rows], marker="o", label=key)
    llm_rows = [r for r in rows if r["llm_accuracy"]]
    if llm_rows:
        axes[0].plot([str(r["dim"]) for r in llm_rows], [float(r["llm_accuracy"]) for r in llm_rows],
                     marker="s", linestyle="--", color="black", label="LLM answer accuracy")
    axes[0].set_xlabel("embedding dimension (64 and 32 are not trained Matryoshka sizes)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Retrieval quality vs dimension")
    axes[0].legend()
    axes[1].plot(xs, [float(r["mean_top1_distance"]) for r in rows], marker="o", label="mean top-1 distance")
    axes[1].plot(xs, [float(r["mean_spread_top10"]) for r in rows], marker="o", label="spread: d(top10) - d(top1)")
    axes[1].set_xlabel("embedding dimension")
    axes[1].set_title("Cosine distances shrink and bunch up at low dims")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "dimensions.png", dpi=140)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
