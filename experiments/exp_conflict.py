"""Experiment D: contradictory sources (re-run of the original Experiment 3, plus a mitigation).

data/pos_doc.pdf and data/neg_doc.pdf describe the same fictional fruit with opposite facts.
Both are ingested into one collection; each question is asked
  (1) with the project's prompt (rag.py), and
  (2) with a conflict-aware prompt that asks the model to report disagreements.
Each question is also asked in a "one word" form, which is where the original run
saw the model silently pick a side.

usage: python experiments/exp_conflict.py
output: results/conflict/results.json, summary.md
"""
import json
import shutil

from exp_dimensions import ROOT, WORK, PROMPT                         # also sets RAG_DB_PATH
from langchain_chroma import Chroma
from loaders import load_file
from chunker import chunk_documents
from models import base_embeddings, llm

OUT = ROOT / "results" / "conflict"

AWARE = ("Answer the question using only the context below. Each passage is labelled with its source file. "
         "If the sources disagree, do NOT pick one: say that they conflict and state what each source claims, "
         "citing the file names. If the answer is not in the context, say you don't know.\n\n"
         "Context:\n{context}\n\nQuestion: {question}")

QUESTIONS = [
    "What does the Aurelian Luma Berry taste like?",
    "What does the Aurelian Luma Berry taste like? Answer in one word.",
    "Is the Luma Berry safe to eat?",
    "Is the Luma Berry safe to eat? Answer yes or no.",
    "What effect does the Luma Berry have on the common cold?",
    "How much water does the Luma Berry need?",
    "What colour is the Luma Berry?",
    "Who discovered the Luma Berry and when?",       # control: both sources AGREE
]

POS = {"sweet", "honey", "vanilla", "safe", "yes", "cure", "boost", "one cup", "yellow", "glowing"}
NEG = {"toxic", "battery", "spoiled", "unsafe", "no", "cause", "suppress", "swamp", "submersion", "gray", "grey", "rotting"}


def sides(ans):
    a = ans.lower()
    words = set(a.replace(".", " ").replace(",", " ").split())
    p = any(k in a if " " in k else k in words for k in POS)
    n = any(k in a if " " in k else k in words for k in NEG)
    return "both" if p and n else "positive only" if p else "negative only" if n else "neither"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    store = Chroma(collection_name="conflict", embedding_function=base_embeddings,
                   persist_directory=str(WORK), collection_metadata={"hnsw:space": "cosine"})
    for f in ["data/pos_doc.pdf", "data/neg_doc.pdf"]:
        chunks = chunk_documents(load_file(f))
        store.add_documents(list(chunks.values()), ids=list(chunks))
    results = []
    for q in QUESTIONS:
        res = store.similarity_search_with_score(q, k=3)
        ctx = "\n\n".join(f"[{d.metadata['source']}, page {d.metadata.get('page', '?')}] {d.page_content}"
                          for d, _ in res)
        row = {"question": q,
               "retrieved": [{"source": d.metadata["source"], "distance": round(s, 4), "text": d.page_content}
                             for d, s in res]}
        for name, tmpl in [("baseline_prompt", PROMPT), ("conflict_aware_prompt", AWARE)]:
            ans = llm.invoke(tmpl.format(context=ctx, question=q)).content.strip()
            row[name] = {"answer": ans, "sides": sides(ans),
                         "mentions_conflict": any(w in ans.lower() for w in
                                                  ["conflict", "contradict", "disagree", "however", "while", "differ"])}
        results.append(row)
        print(json.dumps(row, indent=1, ensure_ascii=False)[:1500])
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["| question | sources in top-3 | baseline: sides in answer | baseline: flags conflict | conflict-aware: sides | conflict-aware: flags conflict |",
          "|---|---|---|:---:|---|:---:|"]
    for r in results:
        srcs = ", ".join(x["source"].replace("_doc.pdf", "") for x in r["retrieved"])
        b, c = r["baseline_prompt"], r["conflict_aware_prompt"]
        md.append(f"| {r['question']} | {srcs} | {b['sides']} | {'yes' if b['mentions_conflict'] else 'no'} | "
                  f"{c['sides']} | {'yes' if c['mentions_conflict'] else 'no'} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
