"""Experiment C: compare the four chunking strategies on the same questions.

Corpus: data/doc_40.pdf. Full 768-d embeddings, TOP_K=3 for the LLM, same 30 questions.
Note: bigger chunks make a "hit" easier (more text per chunk) but cost more LLM context,
so chunk size is reported next to every score.

usage: python experiments/exp_chunkers.py [--no-llm]
output: results/chunkers/summary.csv, summary.md, per_question.json
"""
import argparse
import csv
import json
import shutil
import statistics
import time

from exp_dimensions import ROOT, WORK, Cached, PROMPT, K, TOP_K       # also sets RAG_DB_PATH
from langchain_chroma import Chroma
from loaders import load_file
from chunker import chunk_documents
from models import base_embeddings, llm
from qa_git import QA, is_hit, is_correct

OUT = ROOT / "results" / "chunkers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    docs_in = load_file("data/doc_40.pdf")
    cached = Cached(base_embeddings)
    rows, per_q = [], {}
    for strat in ["paragraph", "recursive", "character", "semantic"]:
        t = time.perf_counter()
        chunks = chunk_documents(docs_in, strat)
        chunk_s = time.perf_counter() - t
        ids = list(chunks)
        docs = [chunks[i] for i in ids]
        t = time.perf_counter()
        store = Chroma(collection_name=f"chunkexp_{strat}", embedding_function=cached,
                       persist_directory=str(WORK), collection_metadata={"hnsw:space": "cosine"})
        store.add_documents(docs, ids=ids)
        embed_s = time.perf_counter() - t
        lens = [len(d.page_content) for d in docs]
        h1 = h3 = 0
        rr, correct, ctx_chars, words = [], 0, [], []
        for item in QA:
            res = store.similarity_search_with_score(item["q"], k=K)
            flags = [is_hit(d.page_content, item) for d, _ in res]
            rank = next((i + 1 for i, f in enumerate(flags) if f), None)
            h1 += any(flags[:1]); h3 += any(flags[:3])
            rr.append(1 / rank if rank else 0.0)
            ctx = "\n\n".join(f"[{d.metadata['source']}, page {d.metadata.get('page', '?')}] {d.page_content}"
                              for d, _ in res[:TOP_K])
            ctx_chars.append(len(ctx))
            entry = {"chunker": strat, "rank_of_evidence": rank,
                     "top3": [{"page": d.metadata.get("page"), "distance": round(s, 4), "hit": f,
                               "text": d.page_content[:160]} for (d, s), f in zip(res[:3], flags[:3])]}
            if not a.no_llm:
                ans = llm.invoke(PROMPT.format(context=ctx, question=item["q"])).content
                ok = is_correct(ans, item)
                correct += ok
                words.append(len(ans.split()))
                entry.update(answer=ans, answer_correct=ok)
            per_q.setdefault(item["q"], []).append(entry)
        n = len(QA)
        row = {"chunker": strat, "chunks": len(docs), "min_chars": min(lens),
               "avg_chars": sum(lens) // len(lens), "max_chars": max(lens),
               "chunk_s": f"{chunk_s:.3f}", "embed_store_s": f"{embed_s:.3f}",
               "hit@1": f"{h1 / n:.3f}", "hit@3": f"{h3 / n:.3f}", "mrr@10": f"{statistics.mean(rr):.3f}",
               "avg_context_chars": int(statistics.mean(ctx_chars)),
               "llm_accuracy": f"{correct / n:.3f}" if words else "",
               "mean_answer_words": f"{statistics.mean(words):.1f}" if words else ""}
        rows.append(row)
        print(row)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "per_question.json").write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["| chunker | chunks | avg / max chars | chunking time (s) | hit@1 | hit@3 | MRR@10 | LLM context chars (top-3) | LLM accuracy |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['chunker']} | {r['chunks']} | {r['avg_chars']} / {r['max_chars']} | {r['chunk_s']} | {r['hit@1']} | "
                  f"{r['hit@3']} | {r['mrr@10']} | {r['avg_context_chars']} | {r['llm_accuracy'] or '-'} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
