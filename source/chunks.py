from typing import List
from unstructured.partition.pdf import partition_pdf


def load_pdf_chunks(path: str, chunk_size: int = 50, overlap: int = 10,) -> List[str]:

    # Parse the PDF into semantic text elements (ignores images and tables)
    elements = partition_pdf(
        filename=path,
        strategy="fast",
        languages=["fr"],
        infer_table_structure=False,
        include_page_breaks=False,
    )

    # Keep only non-empty visible text blocks
    texts: List[str] = [el.text.strip() for el in elements if el.text and el.text.strip()]

    # Chunk text with overlap to preserve context
    chunks: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for block in texts:
        words = block.split()
        block_len = len(words)

        # Flush chunk if size exceeded
        if current_len + block_len > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap > 0 else []
            current_len = len(current_words)

        current_words.extend(words)
        current_len += block_len

    # Append remaining content
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks