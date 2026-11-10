"""Build controlled PDF versions for the incremental re-ingestion experiment.

v0_base    : N paragraphs taken from data/doc_40.pdf (Pro Git, ch. 1-2)
v1_insert  : v0 + ONE new paragraph inserted in the middle
v2_edit    : v1 + ONE word changed inside one paragraph near the end
v3_delete  : v2 - ONE paragraph removed near the start
v4_append  : v3 + ONE paragraph appended at the very end (no reflow of earlier pages)

Every version is rendered with the same layout, so the ONLY difference between
two versions is the edit itself (plus whatever page reflow that edit causes).

usage: python experiments/make_corpus.py [--paragraphs 120] [--out experiments/corpus]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpdf import FPDF                                  # pip install fpdf2
from loaders import load_file
from chunker import _split_paragraphs

ASCII = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-",
                       "–": "-", "•": "-", "…": "..."})

NEW_PARAGRAPH = ("Inserted paragraph for the re-ingestion experiment: Git stores every object "
                 "in a content-addressed database, so two identical files always share one blob. "
                 "This is the same idea our pipeline uses for chunk IDs.")
APPENDED_PARAGRAPH = ("Appended paragraph: a brand new section was added at the very end of the "
                      "document, so no earlier page should reflow.")


def clean(t):
    t = t.translate(ASCII)
    return t.encode("latin-1", "ignore").decode("latin-1")   # core PDF fonts are latin-1 only


def render(paragraphs, path):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for p in paragraphs:
        pdf.multi_cell(0, 5.5, p)
        pdf.ln(5.5)                                    # blank line = paragraph boundary
    pdf.output(str(path))
    return pdf.page_no()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", type=int, default=120)
    ap.add_argument("--out", default="experiments/corpus")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = load_file("data/doc_40.pdf")
    paras = []
    for d in docs:
        paras += [clean(p) for p in _split_paragraphs(d.page_content)]
    paras = [p for p in paras if len(p) >= 60][: a.paragraphs]

    v0 = list(paras)
    mid, late, early = len(v0) // 2, int(len(v0) * 0.75), len(v0) // 4
    v1 = v0[:mid] + [NEW_PARAGRAPH] + v0[mid:]
    v2 = list(v1)
    words = v2[late + 1].split(" ")
    words[len(words) // 2] = "EDITED"
    v2[late + 1] = " ".join(words)
    v3 = v2[:early] + v2[early + 1:]
    v4 = v3 + [APPENDED_PARAGRAPH]

    manifest = {}
    for name, ps in [("v0_base", v0), ("v1_insert", v1), ("v2_edit", v2),
                     ("v3_delete", v3), ("v4_append", v4)]:
        pages = render(ps, out / f"{name}.pdf")
        manifest[name] = {"paragraphs": len(ps), "pages": pages}
        print(f"{name:<10} {len(ps):>4} paragraphs  {pages:>3} pages")
    manifest["edits"] = {
        "v1_insert": f"new paragraph inserted at position {mid}: {NEW_PARAGRAPH!r}",
        "v2_edit": f"paragraph {late + 1}: one word replaced with 'EDITED'",
        "v3_delete": f"paragraph {early} removed",
        "v4_append": f"paragraph appended at the end: {APPENDED_PARAGRAPH!r}",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
