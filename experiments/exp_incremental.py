"""Experiment A: incremental (content-hash diff) vs full re-ingestion.

Drives the REAL CLI (`python main.py ingest ...`) in a throw-away ChromaDB, so the
numbers come from exactly the code path a user runs. The model warm-up happens
before the timer starts (see main.ingest), so each row is pure load+chunk+embed+store.

For every chunker and repetition the same edit sequence is replayed:
  v0 full (baseline) -> v0 incremental (no change)
  -> v1 insert  (incremental, then full)
  -> v2 edit    (incremental, then full)
  -> v3 delete  (incremental, then full)
  -> v4 append  (incremental, then full)

usage: python experiments/exp_incremental.py [--reps 3] [--chunkers paragraph recursive character semantic]
output: results/incremental/raw_timings.csv, summary.csv, summary.md, *.png
"""
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CORPUS = ROOT / "experiments" / "corpus"
OUT = ROOT / "results" / "incremental"
WORK = ROOT / "experiments" / "_work"

STEPS = [  # (label, version file, full?)
    ("base-full", "v0_base", True),
    ("no-change-inc", "v0_base", False),
    ("insert-inc", "v1_insert", False),
    ("insert-full", "v1_insert", True),
    ("edit-inc", "v2_edit", False),
    ("edit-full", "v2_edit", True),
    ("delete-inc", "v3_delete", False),
    ("delete-full", "v3_delete", True),
    ("append-inc", "v4_append", False),
    ("append-full", "v4_append", True),
]


def run_chain(chunker, rep, timings_csv, log):
    db = WORK / f"db_{chunker}_{rep}"
    shutil.rmtree(db, ignore_errors=True)
    target = WORK / "corpus.pdf"                       # same filename every time = same document identity
    env = dict(os.environ, RAG_DB_PATH=str(db), RAG_TIMINGS_CSV=str(timings_csv),
               PYTHONIOENCODING="utf-8")
    for label, version, full in STEPS:
        shutil.copyfile(CORPUS / f"{version}.pdf", target)
        cmd = [PY, "main.py", "ingest", str(target), "--chunker", chunker,
               "--label", f"{label}|rep{rep}"] + (["--full"] if full else [])
        res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        log.write(f"$ {' '.join(cmd[1:])}\n{res.stdout}{res.stderr}\n")
        if res.returncode:
            raise SystemExit(f"ingest failed:\n{res.stderr}")
        print(f"  {chunker:<10} rep{rep} {label:<14}", res.stdout.strip().splitlines()[-3])
    shutil.rmtree(db, ignore_errors=True)


def summarise(raw_csv):
    rows = list(csv.DictReader(open(raw_csv, encoding="utf-8")))
    groups = {}
    for r in rows:
        label = r["label"].split("|")[0]
        groups.setdefault((r["chunker"], label), []).append(r)
    out = []
    for (chunker, label), rs in groups.items():
        med = lambda k: statistics.median(float(r[k]) for r in rs)
        out.append({
            "chunker": chunker, "step": label, "runs": len(rs),
            "total_chunks": rs[0]["total_chunks"], "embedded": rs[0]["embedded"],
            "deleted": rs[0]["deleted"], "skipped": rs[0]["skipped"],
            "pct_reembedded": f"{100 * int(rs[0]['embedded']) / int(rs[0]['total_chunks']):.1f}",
            "median_chunk_s": f"{med('chunk_s'):.3f}",
            "median_embed_store_s": f"{med('embed_store_s'):.3f}",
            "median_total_s": f"{med('total_s'):.3f}",
            "min_total_s": f"{min(float(r['total_s']) for r in rs):.3f}",
            "max_total_s": f"{max(float(r['total_s']) for r in rs):.3f}",
        })
    order = {s[0]: i for i, s in enumerate(STEPS)}
    out.sort(key=lambda r: (r["chunker"], order[r["step"]]))
    return out


def write_outputs(summary):
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    by = {(r["chunker"], r["step"]): r for r in summary}
    chunkers = sorted({r["chunker"] for r in summary})
    lines = ["| chunker | edit | chunks in file | incremental: embedded | incremental: median s | full: median s | speed-up |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c in chunkers:
        base = by[(c, "base-full")]
        nc = by[(c, "no-change-inc")]
        lines.append(f"| {c} | none | {nc['total_chunks']} | {nc['embedded']} | {nc['median_total_s']} | "
                     f"{base['median_total_s']} | {float(base['median_total_s']) / float(nc['median_total_s']):.1f}x |")
        for e in ["insert", "edit", "delete", "append"]:
            inc, full = by[(c, f"{e}-inc")], by[(c, f"{e}-full")]
            lines.append(f"| {c} | {e} | {inc['total_chunks']} | {inc['embedded']} ({inc['pct_reembedded']}%) | "
                         f"{inc['median_total_s']} | {full['median_total_s']} | "
                         f"{float(full['median_total_s']) / float(inc['median_total_s']):.1f}x |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edits = ["no-change", "insert", "edit", "delete", "append"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.8 / len(chunkers)
    for i, c in enumerate(chunkers):
        inc = [float(by[(c, f"{e}-inc")]["median_total_s"]) for e in edits]
        full = [float(by[(c, "base-full" if e == "no-change" else f"{e}-full")]["median_total_s"]) for e in edits]
        pct = [float(by[(c, f"{e}-inc")]["pct_reembedded"]) for e in edits]
        xs = [x + i * w for x in range(len(edits))]
        axes[0].bar(xs, full, w, color=f"C{i}", alpha=0.35)
        axes[0].bar(xs, inc, w, color=f"C{i}", label=c)
        axes[1].bar(xs, pct, w, color=f"C{i}", label=c)
    for ax in axes:
        ax.set_xticks([x + w * (len(chunkers) - 1) / 2 for x in range(len(edits))])
        ax.set_xticklabels(edits)
        ax.legend()
    axes[0].set_ylabel("seconds (solid = incremental, faded = full rebuild)")
    axes[0].set_title("Re-ingest time after a one-paragraph edit (median)")
    axes[1].set_ylabel("% of chunks re-embedded")
    axes[1].set_title("How much of the file an edit invalidates")
    plt.tight_layout()
    plt.savefig(OUT / "incremental_vs_full.png", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunkers", nargs="+", default=["paragraph", "recursive", "character", "semantic"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw_timings.csv"
    raw.unlink(missing_ok=True)
    with open(OUT / "run_log.txt", "w", encoding="utf-8") as log:
        for c in a.chunkers:
            for rep in range(1, a.reps + 1):
                run_chain(c, rep, raw, log)
    summary = summarise(raw)
    write_outputs(summary)
    shutil.copyfile(CORPUS / "manifest.json", OUT / "corpus_manifest.json")


if __name__ == "__main__":
    main()
