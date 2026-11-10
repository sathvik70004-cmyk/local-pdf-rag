from pathlib import Path
from .pdf_loader import load_pdf

LOADERS = {
    ".pdf": load_pdf,                 # later: ".mp3": load_audio, ".png": load_image
}


def load_file(path):
    ext = Path(path).suffix.lower()   # ".pdf" from "data/doc.pdf"
    if ext not in LOADERS:
        raise ValueError(f"No loader for {ext} files yet")
    return LOADERS[ext](path)         # call the matching loader