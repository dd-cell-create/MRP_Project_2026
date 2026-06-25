"""
query_ollama.py (FAST VERSION)
==============================
- Mistral only (fastest local model)
- Context window cut to 512 tokens
- Shortest possible prompt
- No delays between questions
"""

import argparse
import sys
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "mistral"

PROMPT = """Answer this Canadian investor question in 2-3 sentences. Be specific with numbers/rules.

Q: {question}
A:"""


def check_ollama():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        sys.exit("\n❌ Ollama not running. Run: ollama serve\n")
    print(f"✅ Ollama running")


def ollama_call(prompt: str) -> str:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 512,
            "num_predict": 150,
        },
    }
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="final_150_random.csv")
    parser.add_argument("--output",     default="ollama_responses.csv")
    parser.add_argument("--checkpoint", default="ollama_checkpoint.csv")
    args = parser.parse_args()

    check_ollama()

    df = pd.read_csv(args.input)
    df = df[df["label"] == "factual"].reset_index(drop=True)
    print(f"✅ {len(df)} factual questions")

    # resume
    done_ids, results = set(), []
    try:
        ck       = pd.read_csv(args.checkpoint)
        results  = ck.to_dict("records")
        done_ids = set(ck["final_question_id"].tolist())
        print(f"✅ Resuming — {len(done_ids)} done, {len(df)-len(done_ids)} remaining")
    except FileNotFoundError:
        pass

    total = len(df)
    start = time.time()

    for idx, row in df.iterrows():
        qid      = row["final_question_id"]
        question = str(row["cleaned_question"]).strip()

        if qid in done_ids:
            continue

        t0       = time.time()
        response = ollama_call(PROMPT.format(question=question))
        elapsed  = time.time() - t0
        total_elapsed = time.time() - start
        remaining = total - (idx + 1)
        avg = total_elapsed / max(idx + 1 - len(done_ids), 1)
        eta = avg * remaining

        print(f"[{idx+1}/{total}] {qid} ({elapsed:.1f}s) ETA: {eta/60:.1f}min — {question[:45]}...")

        results.append({
            "final_question_id": qid,
            "label":             row.get("label", ""),
            "cleaned_question":  question,
            "mistral_response":  response,
        })
        done_ids.add(qid)

        # checkpoint every 5 questions
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(args.checkpoint, index=False)

    pd.DataFrame(results).to_csv(args.checkpoint, index=False)
    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    print("\n" + "="*50)
    print(f"COMPLETE — {len(out)} questions")
    print(f"Output: {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()


