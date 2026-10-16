from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}

def load_documents(directory: str | Path) -> list[dict]:
    """Load Markdown/text files while preserving source and title metadata."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append({
                    "text": text,
                    "source": path.as_posix(),
                    "title": path.stem.replace("_", " ").replace("-", " ").title(),
                })
    return documents

def validate_documents(documents):
    clean = []
    for doc in documents:
        text = str(doc.get("text", "")).strip()
        if text:
            clean.append({
                "text": text,
                "source": str(doc.get("source", "unknown")),
                "title": str(doc.get("title", "Untitled")),
            })
    return clean
