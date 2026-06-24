"""
02_criteria_based_judge_groq.py
================================
Criteria-based hallucination detection using Groq API (free, fast).
Processes ollama-knowledge rows from ground_truth_updated.csv.

For each question:
  - Runs 5 times on Mixtral (via Groq)
  - Runs 5 times on Llama3 (via Groq)
  - Judge evaluates each answer against 5-category hallucination taxonomy
  - Majority vote → final verdict + confidence score

Groq free tier: 14,400 requests/day — more than enough for 90 questions.
Expected runtime: ~5-10 minutes total.

Output:
  criteria_based_results.csv
  criteria_based_summary.csv
"""

import argparse
import json
import re
import sys
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ▶▶ PASTE YOUR GROQ API KEY HERE
GROQ_API_KEY = "paste-your-key-here"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODELS      = ["mistral", "llama3.1"]
JUDGE_MODEL = "mistral"
N_RUNS      = 1

ANSWER_PROMPT = """You are an AI assistant answering a question from a Canadian retail investor.
Provide a helpful, specific answer. Include any relevant Canadian rules, account types,
platforms, or figures where appropriate.

Question: {question}

Answer:"""

JUDGE_PROMPT = """You are a hallucination detector evaluating an AI response to a Canadian retail investment question.
Use your knowledge of Canadian financial regulations to identify errors. No external reference needed.

Question: {question}

AI Answer: \"\"\"{answer}\"\"\"

Check the answer for any of these 5 hallucination types:
1. fabricated_facts — made-up numbers, statistics, return figures, or claims with no basis
2. regulatory_error — wrong Canadian rules e.g. incorrect TFSA/RRSP/FHSA contribution limits, wrong tax rules
3. non_existent_product — platforms, funds, ETFs, or accounts that do not exist in Canada
4. misleading_risk — dangerous or grossly incorrect risk characterizations
5. outdated_information — rules or figures no longer current in Canada

Mark hallucinated as "yes" if the answer contains ANY of these.
Mark "no" if reasonable with no fabricated or incorrect Canadian financial facts.
Mark "uncertain" if you cannot determine without more context.

Reply ONLY with valid JSON, no markdown:
{{"hallucinated": "yes/no/uncertain", "hallucination_type": "one of the 5 types or none", "rationale": "one sentence"}}"""


def check_api():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    except Exception:
        sys.exit("\n❌ Ollama not running. Run: ollama serve\n")
    installed = {m["name"].split(":")[0] for m in r.get("models", [])}
    missing = [m for m in MODELS + [JUDGE_MODEL] if m.split(":")[0] not in installed]
    if missing:
        sys.exit(f"\n❌ Missing models: {missing}\n")
    print(f"✅ Ollama running — models: {installed}")


def groq_call(model: str, prompt: str, temperature: float = 0.7,
              max_retries: int = 3) -> str:
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_ctx": 1024},
    }
    delay = 3.0
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=body, timeout=180)
            r.raise_for_status()
            return r.json()["response"].strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                return f"ERROR: {exc}"
            time.sleep(delay); delay *= 2
    return "ERROR"


def parse_judge(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {
        "hallucinated":       "uncertain",
        "hallucination_type": "none",
        "rationale":          f"parse failed: {text[:120]}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="ground_truth_updated.csv")
    parser.add_argument("--output",     default="criteria_based_results.csv")
    parser.add_argument("--summary",    default="criteria_based_summary.csv")
    parser.add_argument("--checkpoint", default="criteria_checkpoint.csv")
    args = parser.parse_args()

    check_api()

    df = pd.read_csv(args.input)
    target = df[df["source_type"] == "ollama-knowledge"].reset_index(drop=True)
    print(f"✅ {len(target)} ollama-knowledge questions to process")

    # resume
    done, results = set(), []
    try:
        ck      = pd.read_csv(args.checkpoint)
        results = ck.to_dict("records")
        done    = set(zip(ck["final_question_id"], ck["model"], ck["attempt"]))
        print(f"✅ Resuming — {len(done)} runs already done")
    except FileNotFoundError:
        pass

    total_q = len(target)
    for idx, row in target.iterrows():
        qid      = row["final_question_id"]
        question = str(row["cleaned_question"]).strip()

        print(f"\n[{idx+1}/{total_q}] {qid} — {question[:65]}...")

        for model in MODELS:
            short = model.split("-")[0]
            print(f"  🤖 {short} ({N_RUNS} runs):", end=" ", flush=True)

            for attempt in range(1, N_RUNS + 1):
                if (qid, model, attempt) in done:
                    print(f"{attempt}(skip)", end=" ", flush=True)
                    continue

                answer = groq_call(model, ANSWER_PROMPT.format(question=question),
                                   temperature=0.7)
                judge_raw = groq_call(JUDGE_MODEL,
                                      JUDGE_PROMPT.format(question=question,
                                                          answer=answer[:1200]),
                                      temperature=0.0)
                verdict = parse_judge(judge_raw)
                h = verdict.get("hallucinated", "?")
                print(f"{attempt}:{h[0]}", end=" ", flush=True)

                results.append({
                    "final_question_id":  qid,
                    "cleaned_question":   question,
                    "model":              model,
                    "attempt":            attempt,
                    "answer":             answer,
                    "hallucinated":       verdict.get("hallucinated", "uncertain"),
                    "hallucination_type": verdict.get("hallucination_type", "none"),
                    "judge_rationale":    verdict.get("rationale", ""),
                })
                done.add((qid, model, attempt))

            print()
            pd.DataFrame(results).to_csv(args.checkpoint, index=False)

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    # summary
    summary_rows = []
    for qid in out["final_question_id"].unique():
        q_df     = out[out["final_question_id"] == qid]
        question = q_df["cleaned_question"].iloc[0]
        for model in MODELS:
            m_df = q_df[q_df["model"] == model]
            if len(m_df) == 0:
                continue
            yes = (m_df["hallucinated"] == "yes").sum()
            no  = (m_df["hallucinated"] == "no").sum()
            unc = (m_df["hallucinated"] == "uncertain").sum()
            if yes > N_RUNS / 2:
                verdict = "yes"
            elif no > N_RUNS / 2:
                verdict = "no"
            else:
                verdict = "uncertain"
            types    = m_df[m_df["hallucinated"] == "yes"]["hallucination_type"]
            top_type = types.mode()[0] if len(types) > 0 else "none"
            summary_rows.append({
                "final_question_id":        qid,
                "cleaned_question":         question,
                "model":                    model,
                "runs":                     N_RUNS,
                "hallucinated_count":       yes,
                "not_hallucinated_count":   no,
                "uncertain_count":          unc,
                "hallucination_confidence": f"{round(yes/N_RUNS*100,1)}%",
                "final_verdict":            verdict,
                "top_hallucination_type":   types.mode()[0] if len(types) > 0 else "none",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary, index=False)

    print("\n" + "="*60)
    print("CRITERIA-BASED HALLUCINATION DETECTION COMPLETE")
    print("="*60)
    print(f"Questions: {len(target)}  |  Total runs: {len(out)}")
    print()
    for model in MODELS:
        m_sum = summary_df[summary_df["model"] == model]
        if len(m_sum) == 0: continue
        yes = (m_sum["final_verdict"] == "yes").sum()
        no  = (m_sum["final_verdict"] == "no").sum()
        unc = (m_sum["final_verdict"] == "uncertain").sum()
        n   = len(m_sum)
        print(f"{model.split('-')[0]:10} hallucinated={yes}/{n} ({yes/n:.1%})  "
              f"no={no}  uncertain={unc}")
    print(f"\nResults : {args.output}")
    print(f"Summary : {args.summary}")
    print("="*60)


if __name__ == "__main__":
    main()
