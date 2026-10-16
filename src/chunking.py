def chunk_documents(documents, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Split text into overlapping character windows and retain source metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for doc in documents:
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            part = text[start:start + chunk_size].strip()
            if part:
                chunks.append({
                    "text": part,
                    "source": str(doc.get("source", "unknown")),
                    "title": str(doc.get("title", "Untitled")),
                    "chunk_id": len(chunks),
                    "start_char": start,
                })
            if start + chunk_size >= len(text):
                break
            start += step
    return chunks
