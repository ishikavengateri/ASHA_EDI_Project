"""
test_chromadb.py
Checks that ChromaDB works with the 'nomic-embed-text' embedding model from
Ollama: creates a database folder, stores a few sample sentences, searches
them by meaning, and cleans up afterwards.

Before running (once):
    ollama pull nomic-embed-text

How to run (in Terminal, inside the ASHA_EDI_FY_Project folder):
    source venv/bin/activate
    python test_chromadb.py

What is an "embedding"?
    A list of numbers that represents the meaning of a sentence.
    Sentences with similar meanings get similar numbers, so ChromaDB can
    find "fever in babies" even if the stored text says "high temperature
    in a newborn". A smaller "distance" means a closer match.

Here, Ollama (running on your computer) makes the embeddings with
nomic-embed-text, and ChromaDB stores and searches them. Fully offline.
"""

import sys
import time

import chromadb
import ollama

DB_FOLDER = "chroma_db"              # ChromaDB saves its files in this folder
TEST_COLLECTION = "connection_test"  # temporary collection, deleted at the end
EMBED_MODEL = "nomic-embed-text"

SAMPLE_DOCS = [
    "A newborn with a high temperature or fever should be referred to a health facility immediately.",
    "Exclusive breastfeeding is recommended for the first six months of a baby's life.",
    "ORS and zinc should be given to a child who has diarrhoea.",
    "Iron and folic acid tablets help prevent anaemia in pregnant women.",
]


def embed(texts, kind):
    """
    Turn a list of texts into embeddings using nomic-embed-text.
    nomic-embed-text works best when each text starts with a label:
      'search_document: ' for text we store, 'search_query: ' for questions.
    """
    prefix = "search_document: " if kind == "document" else "search_query: "
    response = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return response.embeddings


def check_embedding_model():
    """Step 1: can Ollama make embeddings with nomic-embed-text?"""
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


def main():
    check_embedding_model()

    # 2) Connect (this creates the chroma_db folder if it doesn't exist)
    print("\n2) Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=DB_FOLDER)
        client.heartbeat()   # simple "are you alive?" check
    except Exception as error:
        print("   FAILED:", error)
        sys.exit(1)
    print(f"   OK - ChromaDB version {chromadb.__version__}, data folder '{DB_FOLDER}'")

    existing = [c.name for c in client.list_collections()]
    print(f"   Existing collections: {existing if existing else 'none yet'}")

    # 3) Create a temporary collection (like a table in a normal database)
    print("\n3) Creating a test collection...")
    if TEST_COLLECTION in existing:
        client.delete_collection(TEST_COLLECTION)   # start fresh
    collection = client.create_collection(TEST_COLLECTION)
    print(f"   OK - created '{TEST_COLLECTION}'")

    try:
        # 4) Embed the sample sentences with nomic-embed-text and store them
        print("\n4) Embedding and adding sample documents...")
        start = time.time()
        collection.add(
            ids=[f"doc-{i}" for i in range(len(SAMPLE_DOCS))],
            documents=SAMPLE_DOCS,                       # the original text
            embeddings=embed(SAMPLE_DOCS, "document"),   # the numbers from Ollama
            metadatas=[{"source": "test", "number": i} for i in range(len(SAMPLE_DOCS))],
        )
        print(f"   OK - stored {collection.count()} documents in {time.time() - start:.1f} seconds")

        # 5) Search by meaning. The question uses different words from the stored text.
        question = "What to do if a baby is very hot?"
        print(f"\n5) Searching for: '{question}'")
        results = collection.query(
            query_embeddings=embed([question], "query"),  # embed the question the same way
            n_results=2,
        )

        # Results come back as lists inside lists (one inner list per question)
        for rank, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0]), start=1
        ):
            print(f"   Match {rank} (distance {distance:.3f}): {doc}")

        best = results["documents"][0][0]
        if "fever" in best:
            print("   OK - the best match is the fever sentence, as expected")
        else:
            print("   WARNING - search worked, but the best match was unexpected")

    finally:
        # 6) Clean up so the test leaves nothing behind
        print("\n6) Deleting the test collection...")
        client.delete_collection(TEST_COLLECTION)
        print("   OK")

    print(f"\nAll checks passed. ChromaDB + {EMBED_MODEL} are ready for the ASHA assistant.")


if __name__ == "__main__":
    main()
