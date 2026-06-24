"""
master_clean_dataset.py  (v2)
=============================
Stage 0 of dataset construction: cheap RULE-BASED cleaning of the raw
master file. No API calls. Output feeds into screen_master.py.

Changes from v1:
  - Greedy substring filters fixed: '%removed%' / '%deleted%' now match
    Reddit's literal [removed]/[deleted] markers instead of killing any
    question containing those words (e.g. "money removed from a TFSA").
  - Junk patterns tightened to page-furniture phrases rather than bare
    topic words ('%privacy%' would have removed legitimate questions
    about financial data privacy).
  - The 50-per-source random sampling half is REMOVED. Sampling by
    source cannot produce the domain-stratified design the MRP needs
    (it is why market_data_claims ended up with n=2). Selection is now
    handled by select_150_stratified.py AFTER the LLM screen assigns
    domains to all questions.

Pipeline:
  master_clean_dataset.py  -> master_clean_questions.csv   (this script)
  screen_master.py         -> master_screened.csv          (LLM quality screen)
  select_150_stratified.py -> final_150_v2.csv             (stratified pick)
  classify_and_audit_questions.py -> audited_150.csv       (dual-model labels)

Usage:
    python master_clean_dataset.py
"""

import sqlite3

import pandas as pd

INPUT_FILE = "master_raw_questions_all_sources.csv"
OUTPUT_CLEAN = "master_clean_questions.csv"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()

required_cols = ["question", "source"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["question"] = df["question"].astype(str).str.strip()
df["source"] = df["source"].astype(str).str.strip()
if "category_candidate" in df.columns:
    df["category_candidate"] = df["category_candidate"].astype(str).str.strip()

# =========================
# SQL CLEANING (rule-based junk removal)
# =========================
# Filters target page furniture and dead Reddit posts. Patterns are
# phrase-level, not bare words, to avoid removing legitimate finance
# questions that merely contain a flagged word.

conn = sqlite3.connect(":memory:")
df.to_sql("raw_questions", conn, index=False, if_exists="replace")

clean_query = """
SELECT *
FROM raw_questions
WHERE question IS NOT NULL
  AND TRIM(question) != ''
  AND LENGTH(TRIM(question)) > 12
  AND (
      LENGTH(TRIM(question)) - LENGTH(REPLACE(TRIM(question), ' ', '')) + 1
  ) >= 3
  -- dead Reddit posts: literal markers only
  AND LOWER(question) NOT LIKE '%[removed]%'
  AND LOWER(question) NOT LIKE '%[deleted]%'
  -- page furniture / community-thread noise: phrase-level patterns
  AND LOWER(question) NOT LIKE '%cookie policy%'
  AND LOWER(question) NOT LIKE '%accept cookies%'
  AND LOWER(question) NOT LIKE '%privacy policy%'
  AND LOWER(question) NOT LIKE '%subscribe to%'
  AND LOWER(question) NOT LIKE '%sign in%'
  AND LOWER(question) NOT LIKE '%sign up%'
  AND LOWER(question) NOT LIKE '%advertisement%'
  AND LOWER(question) NOT LIKE '%weekly thread%'
  AND LOWER(question) NOT LIKE '%daily thread%'
  AND LOWER(question) NOT LIKE '%megathread%'
  AND LOWER(question) NOT LIKE '%post navigation%'
  AND LOWER(question) NOT LIKE '%read more%'
  AND LOWER(question) NOT LIKE '%click here%'
"""

clean_df = pd.read_sql_query(clean_query, conn)
conn.close()

# =========================
# EXACT-DUPLICATE REMOVAL
# =========================
# (Near-duplicates -- e.g. CRA template clones -- are handled later in
# select_150_stratified.py with similarity matching; exact dupes only here.)

clean_df["question_clean_key"] = clean_df["question"].str.lower().str.strip()
before = len(clean_df)
clean_df = clean_df.drop_duplicates(subset=["question_clean_key"])
clean_df = clean_df.drop(columns=["question_clean_key"]).reset_index(drop=True)

clean_df["clean_question_id"] = [f"CQ{i + 1:05d}" for i in range(len(clean_df))]
clean_df.to_csv(OUTPUT_CLEAN, index=False)

# =========================
# REPORT
# =========================

print(f"Input rows:            {len(df)}")
print(f"After rule filters:    {before}  (-{len(df) - before})")
print(f"After exact dedupe:    {len(clean_df)}  (-{before - len(clean_df)})")
print(f"Saved:                 {OUTPUT_CLEAN}")
print("\nRows by source:")
print(clean_df["source"].value_counts())
print("\nNext step:")
print("  python screen_master.py --input master_clean_questions.csv "
      "--output master_screened.csv")