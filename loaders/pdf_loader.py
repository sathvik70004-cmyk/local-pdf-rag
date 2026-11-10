import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def stitch_pages(docs):
    # A page break can cut a sentence in half. If a page ends mid-sentence, move the
    # unfinished sentence to the start of the next page, so it stays in one piece.
    for i in range(len(docs) - 1):
        text = docs[i].page_content.rstrip()
        if not text or text[-1] in ".!?:;\"')\u201d\u2019":     # page ends cleanly: nothing to fix
            continue
        ends = [m.end() for m in re.finditer(r"[.!?][\"')\u201d\u2019]*\s+", text)]   # ends of full sentences
        ends.append(text.rfind("\n\n") + 2 if "\n\n" in text else 0)                  # or after a blank line
        cut = max(ends)                                  # where the unfinished sentence begins
        tail = text[cut:].strip()
        if cut == 0 or len(tail) > 500:                  # no boundary found, or tail too long: leave it
            continue
        docs[i].page_content = text[:cut].rstrip()
        docs[i + 1].page_content = tail + " " + docs[i + 1].page_content.lstrip()
    return docs


def load_pdf(path):
    # extraction_mode="layout" keeps the blank gaps between paragraphs, so the
    # chunkers can see where paragraphs start and end (the default mode loses them).
    docs = PyPDFLoader(path, extraction_mode="layout").load()   # one Document per page
    name = Path(path).name                        # just the filename, e.g. "doc.pdf"
    out = []
    for d in docs:
        if not d.page_content.strip():            # skip blank / image-only pages
            continue
        d.metadata = {                            # replace LangChain's metadata with ours
            "source": name,                       # which file it came from
            "type": "pdf",                        # what kind of file
            "page": d.metadata.get("page", 0) + 1,  # PyPDFLoader counts from 0; humans from 1
        }
        out.append(d)
    return stitch_pages(out)                      # repair sentences cut by page breaks