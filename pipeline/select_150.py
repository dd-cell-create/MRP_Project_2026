"""
select_150.py
=============
Randomly selects 150 usable questions from the screened checkpoint file.
Assigns a final_question_id to each.

Output: final_150_random.csv with columns:
  final_question_id, question, cleaned_question, usable
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="checkpoint_v2.csv")
    parser.add_argument("--output", default="final_150_random.csv")
    parser.add_argument("--n",      type=int, default=150)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    # filter to usable only
    usable_df = df[df["usable"] == True].copy()
    print(f"✅ {len(usable_df)} usable questions")

    # drop duplicates on cleaned_question to avoid near-identical questions
    before = len(usable_df)
    usable_df = usable_df.drop_duplicates(subset="cleaned_question").reset_index(drop=True)
    print(f"✅ Dropped {before - len(usable_df)} duplicate cleaned questions")

    if len(usable_df) < args.n:
        print(f"⚠️  Only {len(usable_df)} usable questions available — using all of them")
        sample = usable_df
    else:
        sample = usable_df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    # assign final IDs
    sample = sample.reset_index(drop=True)
    sample.insert(0, "final_question_id", [f"FQ{i+1:05d}" for i in range(len(sample))])

    sample.to_csv(args.output, index=False)

    print("\n" + "="*50)
    print("SELECTION COMPLETE")
    print("="*50)
    print(f"Selected     : {len(sample)} questions")
    print(f"Output       : {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()
