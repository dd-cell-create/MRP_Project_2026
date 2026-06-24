"""
01b_retry_not_found.py
======================
Reads existing ground_truth.csv, finds NOT_FOUND rows,
rephrases each question using Mistral, retries scraping
official sources, then updates ground_truth.csv with results.

Run AFTER 01_get_ground_truth.py.

Prereqs:
  - ollama serve running
  - mistral pulled
  - ground_truth.csv exists (output of 01_get_ground_truth.py)

Output: updates ground_truth.csv in place
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
MODEL      = "mistral"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; MRP-research-bot/1.0)"}

# ── Same pre-mapped URLs as Script 1 ─────────────────────────────────────────
PRE_MAPPED = [
    (["tfsa", "tax-free savings", "tax free savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/tax-free-savings-account.html", "CRA"),
    (["rrsp", "registered retirement savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans.html", "CRA"),
    (["fhsa", "first home savings"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account.html", "CRA"),
    (["resp", "registered education"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/registered-education-savings-plans-resps.html", "CRA"),
    (["capital gain"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains.html", "CRA"),
    (["dividend"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12000-taxable-amount-dividends.html", "CRA"),
    (["cpp", "canada pension plan"],
     "https://www.canada.ca/en/services/benefits/publicpensions/cpp.html", "CRA"),
    (["old age security", " oas "],
     "https://www.canada.ca/en/services/benefits/publicpensions/cpp/old-age-security.html", "CRA"),
    (["guaranteed income supplement", " gis "],
     "https://www.canada.ca/en/services/benefits/publicpensions/cpp/old-age-security/guaranteed-income-supplement.html", "CRA"),
    (["mortgage"],
     "https://www.canada.ca/en/financial-consumer-agency/services/mortgages.html", "FCAC"),
    (["gic", "guaranteed investment certificate"],
     "https://www.getsmarteraboutmoney.ca/investments/gics/", "OSC"),
    (["etf", "exchange traded fund", "vfv", "xeqt", "vgro", "xeqq", "zsp", "cash.to"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/exchange-traded-funds-etfs/", "OSC"),
    (["mutual fund"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/mutual-funds/", "OSC"),
    (["bond"],
     "https://www.getsmarteraboutmoney.ca/investments/bonds/", "OSC"),
    (["margin account", "margin interest", "margin trading"],
     "https://www.getsmarteraboutmoney.ca/investments/stocks/margin-accounts/", "OSC"),
    (["option", "call option", "put option"],
     "https://www.getsmarteraboutmoney.ca/investments/other-investments/options/", "OSC"),
    (["crypto", "bitcoin", "ethereum", "digital asset", "digital token"],
     "https://www.osc.ca/en/investors/investment-products/crypto-assets", "OSC"),
    (["leveraged etf", "leveraged fund"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/exchange-traded-funds-etfs/leveraged-and-inverse-etfs/", "OSC"),
    (["esg", "socially responsible", "halal"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/exchange-traded-funds-etfs/esg-investing/", "OSC"),
    (["power of attorney"],
     "https://www.canada.ca/en/financial-consumer-agency/services/estate-planning/power-of-attorney.html", "FCAC"),
    (["estate"],
     "https://www.canada.ca/en/financial-consumer-agency/services/estate-planning.html", "FCAC"),
    (["wealthsimple", "robo-advisor", "online advisor"],
     "https://www.getsmarteraboutmoney.ca/investments/get-advice/robo-advisers/", "OSC"),
    (["registration categor", "registrant", "advisor registration"],
     "https://www.osc.ca/en/investors/investor-tools/check-registration", "OSC"),
    (["trusted contact"],
     "https://www.osc.ca/en/investors/investor-tools/trusted-contact-person", "OSC"),
    (["behavioral", "behavioural", "loss aversion", "bias"],
     "https://www.getsmarteraboutmoney.ca/learning-path/investing/psychological-traps/", "OSC"),
    (["investment policy statement"],
     "https://www.getsmarteraboutmoney.ca/learning-path/investing/investment-policy-statement/", "OSC"),
    (["inflation", "consumer price index", "cpi"],
     "https://www.bankofcanada.ca/core-functions/monetary-policy/inflation/", "BankOfCanada"),
    (["interest rate", "bank rate", "overnight rate"],
     "https://www.bankofcanada.ca/core-functions/monetary-policy/", "BankOfCanada"),
    (["long-term care", "long term care"],
     "https://www.canada.ca/en/financial-consumer-agency/services/retirement-planning/long-term-care.html", "FCAC"),
    (["travel insurance", "travel medical"],
     "https://www.canada.ca/en/financial-consumer-agency/services/insurance/travel.html", "FCAC"),
    (["small business deduction", "incorporat", "corporation"],
     "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/report-business-income-expenses/claiming-small-business-deduction.html", "CRA"),
    (["lsif", "labour sponsored"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/deductions-credits-expenses/line-41400-labour-sponsored-funds-tax-credit.html", "CRA"),
    (["withholding tax", "rrsp withholding"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans/making-withdrawals/tax-rates-on-withdrawals.html", "CRA"),
    (["overcontribut", "over-contribut"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans/contributions/over-contributed-your-rrsp-prpp.html", "CRA"),
    (["human capital"],
     "https://www.getsmarteraboutmoney.ca/learning-path/investing/human-capital/", "OSC"),
    (["real estate", "reit"],
     "https://www.getsmarteraboutmoney.ca/investments/real-estate/", "OSC"),
    (["stock", "share", "equity"],
     "https://www.getsmarteraboutmoney.ca/investments/stocks/", "OSC"),
    (["workplace savings", "group rrsp", "dpsp", "pension plan"],
     "https://www.getsmarteraboutmoney.ca/saving/workplace-savings-plans/", "OSC"),
    (["sequence of returns", "retirement income", "decumulation"],
     "https://www.getsmarteraboutmoney.ca/learning-path/retirement/sequence-of-returns-risk/", "OSC"),
    (["tax", "income tax", "tax return", "tax refund"],
     "https://www.canada.ca/en/revenue-agency/services/tax/individuals.html", "CRA"),
    (["car loan", "auto loan", "vehicle loan"],
     "https://www.canada.ca/en/financial-consumer-agency/services/loans/auto-loans.html", "FCAC"),
    (["credit", "credit card", "credit score"],
     "https://www.canada.ca/en/financial-consumer-agency/services/credit-cards.html", "FCAC"),
    (["bank account", "chequing", "savings account"],
     "https://www.canada.ca/en/financial-consumer-agency/services/bank-accounts.html", "FCAC"),
    (["expense", "budget", "saving"],
     "https://www.canada.ca/en/financial-consumer-agency/services/financial-toolkit.html", "FCAC"),
    (["retirement", "retire"],
     "https://www.canada.ca/en/financial-consumer-agency/services/retirement-planning.html", "FCAC"),
    (["index fund", "passive invest", "active invest"],
     "https://www.getsmarteraboutmoney.ca/investments/mutual-funds-and-etfs/exchange-traded-funds-etfs/", "OSC"),
    (["down payment", "downpayment", "home buying", "first home"],
     "https://www.canada.ca/en/financial-consumer-agency/services/mortgages/down-payment.html", "FCAC"),
    (["fraud", "scam", "phishing"],
     "https://www.osc.ca/en/investors/investor-tools/fraud-prevention", "OSC"),
    (["finfluencer", "social media invest", "influencer"],
     "https://www.osc.ca/en/investors/investor-tools/social-media-and-investing", "OSC"),
]

FALLBACK_SEARCHES = [
    ("canada.ca",            "https://www.canada.ca/en/sr/srb.html?q={query}"),
    ("getsmarteraboutmoney.ca", "https://www.getsmarteraboutmoney.ca/?s={query}"),
    ("investopedia.com",     "https://www.investopedia.com/search?q={query}"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────
REPHRASE_PROMPT = """You are helping retrieve ground truth for a Canadian retail investment hallucination study.

The following question returned no results when searched on official Canadian financial websites.
Rephrase it into 3 shorter, simpler search queries that would work better on sites like canada.ca, osc.ca, or getsmarteraboutmoney.ca.
Focus on the core financial concept, not the personal scenario.

Original question: {question}

Reply with exactly 3 rephrased queries, one per line, no numbering, no punctuation:"""

GT_SCRAPE_PROMPT = """Extract a concise factual answer to the question using ONLY the text below.
If the text does not contain enough relevant information, reply exactly: NOT_FOUND

Question: {question}

Page text:
{page_text}

Answer (2-4 sentences, factual, include numbers/limits/rules where present):"""

GT_KNOWLEDGE_PROMPT = """You are a Canadian financial reference providing ground truth for a hallucination study.
Answer the question below factually using your knowledge of Canadian personal finance, tax rules, and securities regulation.
Include specific numbers, rules, limits, or regulatory facts where relevant.
Be concise and factual — this is used as ground truth to judge other AI responses.

Question: {question}

Factual answer (3-5 sentences):"""


# ── Ollama ────────────────────────────────────────────────────────────────────
def check_ollama():
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    except Exception:
        sys.exit("\n❌ Ollama not running. Run: ollama serve\n")
    installed = {m["name"].split(":")[0] for m in tags.get("models", [])}
    if MODEL not in installed:
        sys.exit(f"\n❌ Model '{MODEL}' not pulled. Run: ollama pull {MODEL}\n")
    print(f"✅ Ollama running with {MODEL}")


def ollama_call(prompt: str, max_retries: int = 3) -> str:
    body = {
        "model": MODEL, "prompt": prompt, "stream": False,
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
            time.sleep(delay); delay *= 2
    return "ERROR"


# ── Scraping ──────────────────────────────────────────────────────────────────
def scrape(url: str, max_chars: int = 4000) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:max_chars]
    except Exception as exc:
        return f"SCRAPE_ERROR: {exc}"


def search_first_link(search_url: str, target_domain: str) -> str:
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


def try_scrape_and_extract(question: str, rephrased: str) -> tuple:
    """Try pre-mapped URLs then fallback searches for a rephrased question."""
    q_lower = rephrased.lower()

    # pre-mapped
    for keywords, url, source in PRE_MAPPED:
        if any(kw in q_lower for kw in keywords):
            page_text = scrape(url)
            if not page_text.startswith("SCRAPE_ERROR"):
                extracted = ollama_call(
                    GT_SCRAPE_PROMPT.format(question=question, page_text=page_text[:3500])
                )
                if extracted and "NOT_FOUND" not in extracted and len(extracted) > 20:
                    return extracted.strip(), url, page_text[:400], source
            break

    # fallback site search
    query = re.sub(r"[^\w\s]", "", rephrased)[:60].strip().replace(" ", "+")
    for domain, search_template in FALLBACK_SEARCHES:
        search_url  = search_template.format(query=query)
        result_url  = search_first_link(search_url, domain)
        if not result_url:
            continue
        page_text = scrape(result_url)
        if page_text.startswith("SCRAPE_ERROR"):
            continue
        extracted = ollama_call(
            GT_SCRAPE_PROMPT.format(question=question, page_text=page_text[:3500])
        )
        if extracted and "NOT_FOUND" not in extracted and len(extracted) > 20:
            return extracted.strip(), result_url, page_text[:400], "site-search"

    return "", "", "", ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="ground_truth.csv",
                        help="Existing ground truth CSV from 01_get_ground_truth.py")
    parser.add_argument("--output", default="ground_truth_updated.csv",
                        help="Output file (default: ground_truth_updated.csv)")
    args = parser.parse_args()

    check_ollama()

    df = pd.read_csv(args.input)
    not_found = df[df["ground_truth"] == "NOT_FOUND"].copy()
    already   = df[df["ground_truth"] != "NOT_FOUND"].copy()

    print(f"✅ Loaded {len(df)} rows from {args.input}")
    print(f"   Already found : {len(already)}")
    print(f"   NOT_FOUND     : {len(not_found)} → will retry these")

    if len(not_found) == 0:
        print("\n✅ No NOT_FOUND rows — nothing to do.")
        return

    updated = 0
    ollama_fallback = 0

    for idx, row in not_found.iterrows():
        qid      = row["final_question_id"]
        question = str(row["cleaned_question"]).strip()

        print(f"\n[{idx+1}] {qid} — {question[:65]}...")

        # Step 1 — rephrase into 3 search queries
        print("  ✏️  Rephrasing...", end=" ", flush=True)
        raw_rephrases = ollama_call(REPHRASE_PROMPT.format(question=question))
        rephrases = [r.strip() for r in raw_rephrases.strip().split("\n") if r.strip()][:3]
        print(f"got {len(rephrases)} rephrases")

        gt, ref_url, ref_text, source_type = "", "", "", ""

        # Step 2 — try each rephrased query
        for i, rephrased in enumerate(rephrases):
            print(f"  🔍 Try {i+1}: '{rephrased[:55]}'...", end=" ", flush=True)
            gt, ref_url, ref_text, source_type = try_scrape_and_extract(question, rephrased)
            if gt:
                print(f"✅ [{source_type}]")
                break
            else:
                print("NOT_FOUND")

        # Step 3 — Ollama knowledge fallback
        if not gt:
            print("  🧠 Ollama knowledge fallback...", end=" ", flush=True)
            gt = ollama_call(GT_KNOWLEDGE_PROMPT.format(question=question))
            if gt and not gt.startswith("ERROR"):
                ref_url     = "ollama-knowledge"
                source_type = "ollama-knowledge"
                ollama_fallback += 1
                print("✅")
            else:
                gt          = "NOT_FOUND"
                source_type = "not_found"
                print("❌ still NOT_FOUND")

        # Update the row
        df.at[idx, "ground_truth"]   = gt
        df.at[idx, "reference_url"]  = ref_url
        df.at[idx, "reference_text"] = ref_text
        df.at[idx, "source_type"]    = source_type

        if gt != "NOT_FOUND":
            updated += 1

        # save after each row
        df.to_csv(args.output, index=False)

    # Final summary
    final_found     = (df["ground_truth"] != "NOT_FOUND").sum()
    final_not_found = (df["ground_truth"] == "NOT_FOUND").sum()

    print("\n" + "="*55)
    print("RETRY COMPLETE")
    print("="*55)
    print(f"Previously found  : {len(already)}")
    print(f"Newly found       : {updated}")
    print(f"  via scraping    : {updated - ollama_fallback}")
    print(f"  via Ollama know : {ollama_fallback}")
    print(f"Still NOT_FOUND   : {final_not_found}")
    print(f"Total found       : {final_found}/{len(df)} ({final_found/len(df):.1%})")
    print(f"\nOutput: {args.output}")
    print("="*55)
    print("\nNext: run 02_query_llms.py")


if __name__ == "__main__":
    main()
