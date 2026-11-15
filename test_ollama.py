"""
test_ollama.py
Checks that Ollama is running on this computer and that a model can answer.

How to run (in Terminal, inside the ASHA_EDI_FY_Project folder):
    python3 test_ollama.py              # uses the first model you have installed
    python3 test_ollama.py llama3.2     # or name a specific model

Only uses Python's built-in libraries, so nothing extra to install.
"""

import json
import sys
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434"   # Ollama's default address


def call_ollama(path, payload=None, timeout=120):
    """Send a request to Ollama and return the reply as a Python dict."""
    url = OLLAMA_URL + path
    if payload is None:
        request = urllib.request.Request(url)                      # GET request
    else:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )                                                          # POST request
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_server():
    """Step 1: is Ollama running?"""
    print("1) Checking if Ollama is running...")
    try:
        version = call_ollama("/api/version", timeout=5)
        print(f"   OK - Ollama version {version.get('version', 'unknown')}")
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        print("   FAILED - could not reach Ollama at", OLLAMA_URL)
        print("   Fix: open the Ollama app, or run 'ollama serve' in another Terminal.")
        return False


def list_models():
    """Step 2: which models are downloaded?"""
    print("\n2) Looking for installed models...")
    models = call_ollama("/api/tags", timeout=10).get("models", [])
    if not models:
        print("   No models found.")
        print("   Fix: download one, e.g.  ollama pull llama3.2")
        return []
    for m in models:
        size_gb = m.get("size", 0) / 1e9
        print(f"   - {m['name']}  ({size_gb:.1f} GB)")
    return [m["name"] for m in models]


def test_chat(model):
    """Step 3: ask the model a simple question."""
    print(f"\n3) Asking '{model}' a test question (first run can take a while)...")
    question = "In one sentence, what is an ASHA worker in India?"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,   # wait for the full answer instead of word-by-word
    }
    start = time.time()
    reply = call_ollama("/api/chat", payload, timeout=300)
    seconds = time.time() - start

    answer = reply["message"]["content"].strip()
    print(f"   Question: {question}")
    print(f"   Answer:   {answer}")
    print(f"   Time taken: {seconds:.1f} seconds")

    # Speed in tokens (roughly, word pieces) per second, if Ollama reports it
    tokens = reply.get("eval_count")
    duration_ns = reply.get("eval_duration")
    if tokens and duration_ns:
        print(f"   Speed: {tokens / (duration_ns / 1e9):.1f} tokens/second")


def main():
    if not check_server():
        sys.exit(1)

    installed = list_models()
    if not installed:
        sys.exit(1)

    # Use the model typed after the script name, otherwise the first one found
    model = sys.argv[1] if len(sys.argv) > 1 else installed[0]
    if model not in installed and f"{model}:latest" not in installed:
        print(f"\n   '{model}' is not installed. Fix: ollama pull {model}")
        sys.exit(1)

    try:
        test_chat("qwen3.5:9b")
    except urllib.error.HTTPError as e:
        print("   FAILED - Ollama returned an error:", e.read().decode("utf-8"))
        sys.exit(1)
    except TimeoutError:
        print("   FAILED - the model took too long. Try a smaller model.")
        sys.exit(1)

    print("\nAll checks passed. Ollama is ready for the ASHA assistant.")


if __name__ == "__main__":
    main()
