"""
test_pdf_chromadb.py
End-to-end test of the ASHA assistant's search pipeline:

    PDF in data/  ->  text  ->  overlapping chunks  ->  nomic-embed-text
    embeddings (via Ollama)  ->  stored in ChromaDB  ->  test question
    ->  best matching chunks  ->  delete everything

Before running (once):
    ollama pull nomic-embed-text

How to run (in Terminal, inside the ASHA_EDI_FY_Project folder):
    source venv/bin/activate
    python test_pdf_chromadb.py
    python test_pdf_chromadb.py "How to keep a newborn warm?"   # your own question

Uses PyMuPDF (import pymupdf), chromadb and ollama, all already in your venv.
"""

import re
import shutil
import sys
import time
from pathlib import Path

import chromadb
import pymupdf  # PyMuPDF - reads PDFs
import ollama

# ---------- Settings you can change ----------
DATA_FOLDER = Path("data")              # where your PDFs are
DB_FOLDER = Path("chroma_db_pdf_test")  # temporary database, deleted at the end
COLLECTION_NAME = "pdf_test"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500                        # characters per chunk (about 80-100 words)
OVERLAP = 100                           # characters shared between neighbouring chunks
# Smaller chunks = each chunk is about ONE topic, so its embedding is sharper.
# Run tune_chunks.py to measure which size works best for your PDFs.
BATCH_SIZE = 32                         # chunks sent to Ollama at a time
TOP_RESULTS = 3                         # how many matches to show
MARGIN = 0.15                           # top/bottom 15% of a page = header/footer zone
REPEAT_FRACTION = 0.05                  # margin text on 5%+ of pages = header/footer
DEFAULT_QUESTION = "What should an ASHA do if a newborn has fever?"
# ---------------------------------------------


# ===================== Part 1: PDF -> overlapping chunks =====================

def clean_text(text):
    """Tidy up text from a PDF page."""
    text = re.sub(r"-\n(\w)", r"\1", text)   # "chil-\ndren" -> "children"
    text = re.sub(r"\s+", " ", text)          # newlines/tabs/many spaces -> one space
    return text.strip()


def normalise(text):
    """Make lines comparable across pages: lowercase, every number -> '#'."""
    return re.sub(r"\d+", "#", text.strip().lower())


def is_page_number(text):
    """True for lines like '12', 'Page 12', '- 12 -', 'xii'."""
    t = text.strip().lower()
    return bool(re.fullmatch(r"(page\s*)?[-–—\s]*(\d{1,4}|[ivxlc]{1,6})[-–—\s]*", t))


def in_margin(top, bottom, height, horizontal):
    """Is this line normal (not sideways) text in the top or bottom margin zone?"""
    return horizontal and (bottom < height * MARGIN or top > height * (1 - MARGIN))


def remove_headers_footers(raw_pages):
    """
    raw_pages: list of (page_number, page_height, lines), where each line is
               (top_y, bottom_y, text, is_horizontal). y is measured from the
               top of the page.

    A line counts as a header/footer if it is normal horizontal text, sits in
    the top or bottom margin zone, AND either
      - the same text (ignoring numbers) appears in the margins of many pages, or
      - it is just a page number.
    Returns (list of (page_number, cleaned_text), number_of_lines_removed).
    """
    # Pass 1: count on how many pages each margin line appears
    margin_counts = {}
    for _, height, lines in raw_pages:
        seen = set()
        for top, bottom, text, horizontal in lines:
            if in_margin(top, bottom, height, horizontal):
                seen.add(normalise(text))
        for key in seen:
            margin_counts[key] = margin_counts.get(key, 0) + 1

    min_pages = max(3, int(len(raw_pages) * REPEAT_FRACTION))
    repeated = {
        key for key, count in margin_counts.items()
        if count >= min_pages and len(re.sub(r"[^a-z]", "", key)) >= 4  # needs real words
    }

    # Pass 2: rebuild each page's text without those lines
    pages, removed = [], 0
    for page_number, height, lines in raw_pages:
        kept = []
        for top, bottom, text, horizontal in lines:
            if in_margin(top, bottom, height, horizontal) and (
                normalise(text) in repeated or is_page_number(text)
            ):
                removed += 1
                continue
            if kept and text == kept[-1]:          # same line printed twice (bold effect)
                continue
            kept.append(text)
        text = clean_text("\n".join(kept))
        if text:                                   # skip blank / image-only pages
            pages.append((page_number, text))
    return pages, removed


def extract_pages(pdf_path):
    """Return (list of (page_number, text), header/footer lines removed)."""
    raw_pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            lines = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:         # 0 = text block (skip images)
                    continue
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if text:
                        x0, y0, x1, y1 = line["bbox"]
                        # dir = text direction; (1, 0) means normal left-to-right
                        horizontal = abs(line["dir"][0] - 1) < 0.01
                        lines.append((y0, y1, text, horizontal))
            raw_pages.append((page_index + 1, page.rect.height, lines))
    return remove_headers_footers(raw_pages)


def find_break(text, start, end, chunk_size):
    """Pick a good place to end a chunk: after a full stop if possible, else a space."""
    if end >= len(text):
        return len(text)
    min_end = start + chunk_size // 2              # never make a chunk too short
    for marker in (". ", "? ", "! ", "; ", " "):
        position = text.rfind(marker, min_end, end)
        if position != -1:
            return position + 1
    return end


def make_chunks(pages, source_name, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Join all pages and cut into overlapping chunks, remembering page numbers.

    Overlap example with CHUNK_SIZE = 10 and OVERLAP = 3:
        text    = "ABCDEFGHIJKLMNOPQRS"
        chunk 1 = "ABCDEFGHIJ"
        chunk 2 =        "HIJKLMNOPQ"   <- starts 3 letters back
    so a sentence cut at the edge of one chunk is still whole in the next.
    """
    full_text = ""
    page_starts = []                               # (character position, page number)
    for page_number, text in pages:
        page_starts.append((len(full_text), page_number))
        full_text += text + " "

    def page_at(position):
        current = page_starts[0][1]
        for start_pos, page_number in page_starts:
            if start_pos > position:
                break
            current = page_number
        return current

    chunks = []
    start = 0
    while start < len(full_text):
        end = find_break(full_text, start, start + chunk_size, chunk_size)
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{source_name}-chunk-{len(chunks)}",
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page_start": page_at(start),
                    "page_end": page_at(end - 1),
                },
            })
        if end >= len(full_text):
            break
        next_start = end - overlap                 # step back to create the overlap
        space = full_text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


# ===================== Part 2: embeddings with nomic-embed-text =====================

def embed(texts, kind):
    """
    Turn a list of texts into embeddings (lists of numbers) with nomic-embed-text.
    nomic-embed-text works best with a label at the start:
      'search_document: ' for stored text, 'search_query: ' for questions.
    """
    prefix = "search_document: " if kind == "document" else "search_query: "
    response = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return response.embeddings


def check_embedding_model():
    print(f"1) Checking embedding model '{EMBED_MODEL}'...")
    try:
        vector = embed(["test"], "document")[0]
    except ollama.ResponseError as error:
        print("   FAILED:", error)
        print(f"   Fix: ollama pull {EMBED_MODEL}")
        sys.exit(1)
    except Exception:
        print("   FAILED - could not reach Ollama.")
        print("   Fix: open the Ollama app, or run 'ollama serve' in another Terminal.")
        sys.exit(1)
    print(f"   OK - each embedding is a list of {len(vector)} numbers")


# ===================== Part 3: the test =====================

def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    check_embedding_model()

    # 2) Read the PDFs and make chunks
    print(f"\n2) Reading PDFs from '{DATA_FOLDER}'...")
    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print("   No PDF files found. Put a PDF in the data folder and try again.")
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        pages, removed = extract_pages(pdf_path)
        if not pages:
            print(f"   {pdf_path.name}: no text found (may be a scanned PDF). Skipping.")
            continue
        chunks = make_chunks(pages, pdf_path.name)
        all_chunks.extend(chunks)
        print(f"   {pdf_path.name}: {len(pages)} pages, {removed} header/footer lines removed"
              f" -> {len(chunks)} chunks")

    if not all_chunks:
        print("   No text could be extracted from any PDF.")
        sys.exit(1)

    # 3) Connect to a separate test database (so your real data is never touched)
    print(f"\n3) Creating test database in '{DB_FOLDER}'...")
    client = chromadb.PersistentClient(path=str(DB_FOLDER))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    print("   OK")

    try:
        # 4) Embed the chunks in batches and store them
        print(f"\n4) Embedding {len(all_chunks)} chunks with {EMBED_MODEL} and storing them...")
        start_time = time.time()
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=embed(texts, "document"),
                metadatas=[c["metadata"] for c in batch],
            )
            done = min(i + BATCH_SIZE, len(all_chunks))
            print(f"   {done}/{len(all_chunks)} chunks done", end="\r")
        print(f"\n   OK - {collection.count()} chunks stored in {time.time() - start_time:.1f} seconds")

        # 5) Ask a test question
        print(f"\n5) Test question: '{question}'")
        results = collection.query(
            query_embeddings=embed([question], "query"),
            n_results=TOP_RESULTS,
        )
        for rank in range(len(results["ids"][0])):
            meta = results["metadatas"][0][rank]
            distance = results["distances"][0][rank]
            text = results["documents"][0][rank]
            pages = (f"page {meta['page_start']}" if meta["page_start"] == meta["page_end"]
                     else f"pages {meta['page_start']}-{meta['page_end']}")
            print(f"\n   Match {rank + 1}  (distance {distance:.3f})")
            print(f"   File:  {meta['source']}")
            print(f"   Pages: {pages}")
            print(f"   Text:  {text[:700]}{'...' if len(text) > 700 else ''}")

    finally:
        # 6) Delete everything this test created
        print("\n6) Deleting all test data...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        shutil.rmtree(DB_FOLDER, ignore_errors=True)
        print(f"   OK - collection and '{DB_FOLDER}' folder removed")

    print("\nTest complete. PDF -> chunks -> nomic-embed-text -> ChromaDB -> search all work.")


if __name__ == "__main__":
    main()
