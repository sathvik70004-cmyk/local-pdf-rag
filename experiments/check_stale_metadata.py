"""Side finding: does incremental ingest leave stale page numbers behind?

Chunk IDs hash (source + text) only. When an inserted paragraph pushes later paragraphs
onto a different page, their text (and so their ID) is unchanged, so incremental ingest
SKIPS them and the stored `page` metadata keeps the OLD page number.

Steps: full-ingest v0_base -> incremental-ingest v1_insert -> compare stored page
metadata with the page a fresh chunking of v1 reports.

usage: python experiments/check_stale_metadata.py
output: results/incremental/stale_metadata.txt
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "experiments" / "_work"
DB = WORK / "db_stale"
os.environ["RAG_DB_PATH"] = str(DB)
os.environ["RAG_TIMINGS_CSV"] = str(WORK / "stale_timings.csv")
sys.path.insert(0, str(ROOT))


def main():
    shutil.rmtree(DB, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / "corpus.pdf"
    log = []
    for version, full in [("v0_base", True), ("v1_insert", False)]:
        shutil.copyfile(ROOT / "experiments" / "corpus" / f"{version}.pdf", target)
        cmd = [sys.executable, "main.py", "ingest", str(target), "--chunker", "paragraph"] + (["--full"] if full else [])
        r = subprocess.run(cmd, cwd=ROOT, env=os.environ, capture_output=True, text=True, encoding="utf-8")
        log.append(f"$ ingest {version}{' --full' if full else ''}\n{r.stdout.strip()}")

    from loaders import load_file
    from chunker import chunk_documents
    from store import vectorstore
    fresh = chunk_documents(load_file(str(target)), "paragraph")
    stored = vectorstore.get(include=["metadatas"])
    stored_page = dict(zip(stored["ids"], (m["page"] for m in stored["metadatas"])))
    stale = [(cid, stored_page[cid], d.metadata["page"]) for cid, d in fresh.items()
             if cid in stored_page and stored_page[cid] != d.metadata["page"]]
    log.append(f"\nchunks in v1: {len(fresh)} | chunks whose stored page != real page: {len(stale)}")
    for cid, old, new in stale[:15]:
        log.append(f"  {cid}  stored page {old}  ->  actual page {new} :: {fresh[cid].page_content[:70]}")
    text = "\n".join(log)
    print(text)
    (ROOT / "results" / "incremental").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "incremental" / "stale_metadata.txt").write_text(text + "\n", encoding="utf-8")
    del vectorstore
    shutil.rmtree(DB, ignore_errors=True)


if __name__ == "__main__":
    main()
