"""
01_get_ground_truth.py
======================
Script 1 of 2 — Ground Truth Retrieval (fully local, no API keys)

For each factual question:
  1. Keyword match -> go directly to official CRA/OSC/GetSmarterAboutMoney page
  2. If no keyword match -> search canada.ca or getsmarteraboutmoney.ca
  3. Scrape the page
  4. Mistral (local Ollama) extracts the specific answer from page text

Zero API keys needed. Runs fully locally via Ollama.

Prereqs:
  - ollama serve running in a terminal
  - mistral already pulled (ollama pull mistral)
  - pip install requests pandas beautifulsoup4

Output: ground_truth.csv
Next:   run 02_run_llms_and_judge.py
"""

import argparse
import re
import sys
import time
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "mistral"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MRP-research-bot/1.0)"}

# ── Pre-mapped official URLs ──────────────────────────────────────────────────
PRE_MAPPED = [
    (["tfsa", "tax-free savings", "tax free savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/tax-free-savings-account.html",
     "CRA"),
    (["rrsp", "registered retirement savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans.html",
     "CRA"),
    (["fhsa", "first home savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account.html",
     "CRA"),
    (["resp", "registered education"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/registered-education-savings-plans-resps.html",
     "CRA"),
    (["capital gain"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains.html",
     "CRA"),
    (["dividend"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12000-taxable-amount-dividends.html",
     "CRA"),
    (["cpp", "canada pension plan"],
     "https://www.canada.ca/en/services/benefits/publicpensions/cpp.html",
     "CRA"),
    (["old age security", " oas "],
     "https://www.canada.ca/en/services/benefits/publicpensions/cpp/old-age-security.html",
     "CRA"),
    (["mortgage"],
     "https://www.canada.ca/en/financial-consumer-agency/services/mortgages.html",
     "FCAC"),
    (["gic", "guaranteed investment certificate"],
     "https://www.getsmarteraboutmoney.ca/investments/gics/",
     "OSC-GetSmarterAboutMoney"),
    (["etf", "exchange traded fund", "exchange-traded fund"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/exchange-traded-funds-etfs/",
     "OSC-GetSmarterAboutMoney"),
    (["mutual fund"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/mutual-funds/",
     "OSC-GetSmarterAboutMoney"),
    (["bond"],
     "https://www.getsmarteraboutmoney.ca/investments/bonds/",
     "OSC-GetSmarterAboutMoney"),
    (["margin account", "margin trading"],
     "https://www.getsmarteraboutmoney.ca/investments/stocks/margin-accounts/",
     "OSC-GetSmarterAboutMoney"),
    (["option", "call option", "put option"],
     "https://www.getsmarteraboutmoney.ca/investments/other-investments/options/",
     "OSC-GetSmarterAboutMoney"),
    (["crypto", "bitcoin", "ethereum", "digital asset"],
     "https://www.osc.ca/en/investors/investment-products/crypto-assets",
     "OSC"),
    (["investment risk", "risk tolerance", "risk suitability"],
     "https://www.getsmarteraboutmoney.ca/learning-path/investing/investment-risk/",
     "OSC-GetSmarterAboutMoney"),
    (["stock", "share", "equity"],
     "https://www.getsmarteraboutmoney.ca/investments/stocks/",
     "OSC-GetSmarterAboutMoney"),
    (["real estate", "reit"],
     "https://www.getsmarteraboutmoney.ca/investments/real-estate/",
     "OSC-GetSmarterAboutMoney"),
    (["insurance", "life insurance"],
     "https://www.getsmarteraboutmoney.ca/insurance/",
     "OSC-GetSmarterAboutMoney"),
    (["net asset value", "nav"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/mutual-funds/",
     "OSC-GetSmarterAboutMoney"),
    (["inflation", "consumer price index", "cpi"],
     "https://www.bankofcanada.ca/core-functions/monetary-policy/inflation/",
     "BankOfCanada"),
    (["interest rate", "bank rate", "overnight rate"],
     "https://www.bankofcanada.ca/core-functions/monetary-policy/",
     "BankOfCanada"),
]

# Fallback search URLs
FALLBACK_SEARCHES = [
    "https://www.canada.ca/en/sr/srb.html?q={query}",
    "https://www.getsmarteraboutmoney.ca/?s={query}",
    "https://www.investopedia.com/search?q={query}",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
GT_EXTRACT_PROMPT = """Extract a concise factual answer to the question using ONLY the text below.
If the text does not contain enough relevant information, reply exactly: NOT_FOUND

Question: {question}

Page text:
{page_text}

Answer (2-4 sentences, factual, specific — include numbers/limits/rules where present):"""


# ── Ollama ────────────────────────────────────────────────────────────────────
def check_ollama():
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    except Exception:
        sys.exit("\n❌ Ollama not running. Open terminal and run: ollama serve\n")
    installed = {m["name"].split(":")[0] for m in tags.get("models", [])}
    if JUDGE_MODEL not in installed:
        sys.exit(f"\n❌ Model '{JUDGE_MODEL}' not pulled.\nRun: ollama pull {JUDGE_MODEL}\n")
    print(f"✅ Ollama running with {JUDGE_MODEL}")


def ollama_call(prompt: str, max_retries: int = 3) -> str:
    body = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 2048},
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
            time.sleep(delay)
            delay *= 2
    return "ERROR"


# ── Scraping ──────────────────────────────────────────────────────────────────
def scrape(url: str, max_chars: int = 4000) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:max_chars]
    except Exception as exc:
        return f"SCRAPE_ERROR: {exc}"


def search_and_get_first_link(search_url: str, target_domain: str) -> str:
    """Search a site and return first result URL matching target domain."""
    try:
        r = requests.get(search_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if target_domain in href and href.startswith("http"):
                return href
    except Exception:
        pass
    return ""


# ── Ground truth logic ────────────────────────────────────────────────────────
def get_ground_truth(question: str) -> tuple:
    """Returns (ground_truth, reference_url, reference_text, source_type)"""
    q_lower = question.lower()

    # Step 1 — pre-mapped keyword match
    for keywords, url, source in PRE_MAPPED:
        if any(kw in q_lower for kw in keywords):
            page_text = scrape(url)
            if not page_text.startswith("SCRAPE_ERROR"):
                extracted = ollama_call(
                    GT_EXTRACT_PROMPT.format(
                        question=question,
                        page_text=page_text[:3500]
                    )
                )
                if extracted and "NOT_FOUND" not in extracted and len(extracted) > 15:
                    return extracted.strip(), url, page_text[:400], source
            break  # tried the right page, move to fallback

    # Step 2 — site search fallback
    query = re.sub(r"[^\w\s]", "", question)[:60].strip().replace(" ", "+")

    for search_template in FALLBACK_SEARCHES:
        search_url = search_template.format(query=query)

        # determine target domain
        if "canada.ca" in search_url:
            target = "canada.ca"
        elif "getsmarteraboutmoney" in search_url:
            target = "getsmarteraboutmoney.ca"
        else:
            target = "investopedia.com"

        result_url = search_and_get_first_link(search_url, target)
        if not result_url:
            continue

        page_text = scrape(result_url)
        if page_text.startswith("SCRAPE_ERROR"):
            continue

        extracted = ollama_call(
            GT_EXTRACT_PROMPT.format(
                question=question,
                page_text=page_text[:3500]
            )
        )
        if extracted and "NOT_FOUND" not in extracted and len(extracted) > 15:
            return extracted.strip(), result_url, page_text[:400], "site-search"

    return "NOT_FOUND", "", "", "not_found"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="final_150_random.csv")
    parser.add_argument("--output",       default="ground_truth.csv")
    parser.add_argument("--checkpoint",   default="gt_checkpoint.csv")
    parser.add_argument("--label-filter", default="factual")
    args = parser.parse_args()

    check_ollama()

    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")
    if args.label_filter != "all":
        df = df[df["label"] == args.label_filter].reset_index(drop=True)
        print(f"✅ Filtered to {len(df)} '{args.label_filter}' questions")

    # resume support
    done_ids, results = set(), []
    try:
        ck       = pd.read_csv(args.checkpoint)
        results  = ck.to_dict("records")
        done_ids = set(ck["final_question_id"].tolist())
        print(f"✅ Resuming — {len(done_ids)} already done")
    except FileNotFoundError:
        pass

    total = len(df)
    for idx, row in df.iterrows():
        qid      = row["final_question_id"]
        question = str(row["cleaned_question"]).strip()
        label    = str(row.get("label", "")).strip()

        if qid in done_ids:
            continue

        print(f"\n[{idx+1}/{total}] {qid} — {question[:70]}...")
        gt, ref_url, ref_text, source_type = get_ground_truth(question)
        found = gt not in ("NOT_FOUND", "") and bool(gt)
        print(f"  {'✅' if found else '⚠️ NOT_FOUND'} [{source_type}] {ref_url[:65]}")

        results.append({
            "final_question_id": qid,
            "label":             label,
            "cleaned_question":  question,
            "ground_truth":      gt,
            "reference_url":     ref_url,
            "reference_text":    ref_text,
            "source_type":       source_type,
        })
        done_ids.add(qid)
        pd.DataFrame(results).to_csv(args.checkpoint, index=False)

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    found_n    = (out["ground_truth"] != "NOT_FOUND").sum()
    notfound_n = (out["ground_truth"] == "NOT_FOUND").sum()

    print("\n" + "=" * 55)
    print("GROUND TRUTH RETRIEVAL COMPLETE")
    print("=" * 55)
    print(f"Total          : {len(out)}")
    print(f"Found          : {found_n} ({found_n/len(out):.1%})")
    print(f"NOT_FOUND      : {notfound_n} ({notfound_n/len(out):.1%})")
    print(f"\nSource breakdown:")
    print(out["source_type"].value_counts().to_string())
    print(f"\nOutput saved to: {args.output}")
    print("=" * 55)
    print("\nNext: run 02_run_llms_and_judge.py")


if __name__ == "__main__":
    main()
