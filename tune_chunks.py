"""
tune_chunks.py
Finds the best CHUNK_SIZE and OVERLAP for your PDFs by measuring, not guessing.

For each setting it:
  1. cuts the PDFs into chunks (using the same code as test_pdf_chromadb.py),
  2. makes nomic-embed-text embeddings for every chunk,
  3. asks the test questions below, and checks whether a chunk containing the
     known answer comes back near the top.

Nothing is saved to ChromaDB; everything is kept in memory and thrown away.

How to run (in Terminal, inside the ASHA_EDI_FY_Project folder):
    source venv/bin/activate
    python tune_chunks.py

Scores (higher is better):
    Top-1  = % of questions where the FIRST result contains the answer
    Top-3  = % of questions where the answer is somewhere in the first 3 results
    Score  = average of 1/rank (answer at rank 1 -> 1, rank 2 -> 0.5,
             rank 3 -> 0.33 ... not in top 5 -> 0). Combines both ideas.
"""

import re
import time
from pathlib import Path

import numpy as np

from test_pdf_chromadb import DATA_FOLDER, check_embedding_model, embed, extract_pages, make_chunks

# (chunk size in characters, overlap in characters) to compare
SETTINGS = [
    (300, 60),
    (500, 100),
    (700, 140),
    (1000, 200),
    (1500, 300),
]

# Questions whose answers are in "Handbook for ASHA on Home Based Care for Young Child".
# A result counts as correct if it contains ANY of the answer phrases (case ignored).
# Add your own as you add more PDFs.
TEST_QUESTIONS = [
    ("In which months of the child's age does the ASHA make home visits?",
     ["3rd, 6th, 9th, 12th"]),
    ("How much IFA syrup should be given to a child and how often?",
     ["1 ml"]),
    ("For how many days should zinc be given to a child with diarrhoea?",
     ["14 days"]),
    ("What are the signs of severe dehydration in a child?",
     ["skin pinch goes back very slowly"]),
    ("How can I tell if a child has pneumonia?",
     ["chest indrawing"]),
    ("How do you prepare ORS at home if there is no packet?",
     ["teaspoon of sugar"]),
    ("Should IFA syrup be given to a child who is sick with fever?",
     ["should not be given in children suffering from acute illness"]),
    ("Why do young children become anaemic?",
     ["low iron stores at birth"]),
    ("What are the common side effects after a child is vaccinated?",
     ["slight fever, pain, swelling"]),
    ("How can a mother encourage her child to eat enough food?",
     ["play games"]),
]

TOP_K = 5   # look at the first 5 results


def simplify(text):
    """Lowercase and squash spaces so answer phrases match reliably."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def rank_of_answer(question_vec, chunk_vecs, chunk_texts, answers):
    """Return 1, 2, 3... for where the first correct chunk appears, or None."""
    scores = chunk_vecs @ question_vec                 # cosine similarity (vectors are normalised)
    best = np.argsort(-scores)[:TOP_K]                 # indices of the 5 most similar chunks
    for rank, index in enumerate(best, start=1):
        text = chunk_texts[index]
        if any(simplify(a) in text for a in answers):
            return rank
    return None


def normalise_rows(vectors):
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    check_embedding_model()

    # Read the PDFs once (header/footer removal included)
    print(f"\nReading PDFs from '{DATA_FOLDER}'...")
    documents = []
    for pdf_path in sorted(Path(DATA_FOLDER).glob("*.pdf")):
        pages, _ = extract_pages(pdf_path)
        if pages:
            documents.append((pdf_path.name, pages))
            print(f"   {pdf_path.name}: {len(pages)} pages")
    if not documents:
        print("   No PDFs with text found in the data folder.")
        return

    # Embed the questions once
    question_vecs = normalise_rows(embed([q for q, _ in TEST_QUESTIONS], "query"))

    results = []
    for chunk_size, overlap in SETTINGS:
        print(f"\nTesting chunk size {chunk_size}, overlap {overlap}...")
        chunks = []
        for name, pages in documents:
            chunks += make_chunks(pages, name, chunk_size=chunk_size, overlap=overlap)
        texts = [c["text"] for c in chunks]

        start = time.time()
        vectors = []
        for i in range(0, len(texts), 32):
            vectors += embed(texts[i:i + 32], "document")
            print(f"   embedded {min(i + 32, len(texts))}/{len(texts)} chunks", end="\r")
        chunk_vecs = normalise_rows(vectors)
        simple_texts = [simplify(t) for t in texts]
        print(f"   embedded {len(texts)} chunks in {time.time() - start:.0f} s       ")

        ranks = []
        for (question, answers), q_vec in zip(TEST_QUESTIONS, question_vecs):
            rank = rank_of_answer(q_vec, chunk_vecs, simple_texts, answers)
            ranks.append(rank)
            print(f"   {'rank ' + str(rank) if rank else 'MISSED':>7}  {question}")

        n = len(ranks)
        results.append({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": len(chunks),
            "top1": 100 * sum(r == 1 for r in ranks) / n,
            "top3": 100 * sum(r is not None and r <= 3 for r in ranks) / n,
            "score": sum(1 / r for r in ranks if r) / n,
        })

    # Summary table
    print("\n" + "=" * 62)
    print(f"{'Chunk size':>10} {'Overlap':>8} {'Chunks':>7} {'Top-1':>7} {'Top-3':>7} {'Score':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['chunk_size']:>10} {r['overlap']:>8} {r['chunks']:>7} "
              f"{r['top1']:>6.0f}% {r['top3']:>6.0f}% {r['score']:>7.2f}")
    print("=" * 62)

    best = max(results, key=lambda r: (r["score"], r["top3"]))
    print(f"\nBest setting: CHUNK_SIZE = {best['chunk_size']}, OVERLAP = {best['overlap']}")
    print("Put these numbers in the settings at the top of test_pdf_chromadb.py.")


if __name__ == "__main__":
    main()
