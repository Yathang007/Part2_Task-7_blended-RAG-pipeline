# FAISS vector similarity search

FAISS is a library for efficient similarity search over dense vectors. An embedding model maps text to numeric vectors so semantically similar passages can be compared. For cosine similarity, vectors can be normalized to unit length and searched with inner product.

A simple exact index such as IndexFlatIP is a transparent baseline for small corpora. It stores vectors in memory and searches all indexed vectors, so search cost grows with corpus size. Larger collections may benefit from approximate indexes, but those require tuning and evaluation.

Vector similarity is not proof that a passage answers a question. Combine semantic retrieval with lexical retrieval when exact terms, error codes, package names, or identifiers matter. Evaluate retrieval using labeled relevant documents rather than judging only by a few examples.
