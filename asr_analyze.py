#!/usr/bin/env python3
"""
Compute Attack Success Rate (ASR) per model / attack / attack-parameter
combination from a results CSV produced by ML security attack testing runs.

ASR for a given configuration (a unique combo of model + attack + all its
hyperparameters) is defined as:

    ASR = (number of runs with success == True) / (total runs for that config)

Usage:
    python analyze_asr.py results.csv
    python analyze_asr.py results.csv --out asr_summary.csv
    python analyze_asr.py results.csv --min-runs 3
    python analyze_asr.py results.csv --attack fc
"""

import argparse
import sys
import pandas as pd

# Columns that include the outcomes, we want to avoid grouping by these when computing ASR.
OUTCOMES_COLS = {"run", "success", "clean_acc", "poisoned_acc"}

# Computes the ASR per model/attack/params combination, returning a summary DataFrame
def compute_asr(df: pd.DataFrame, min_runs: int = 1) -> pd.DataFrame:
    # Groups columns for ASR computation, excluding outcomes columns.
    all_group_cols = [c for c in df.columns if c not in OUTCOMES_COLS]

    print(f"Computing ASR with grouping columns: {all_group_cols}")

    print("Grouping by attack...")
    
    df.groupby("attack", dropna=False)
    print(df.groupby("attack", dropna=False))

    results = []

    # First we group by attack
    for attack, sub in df.groupby("attack", dropna=False):

        print(f"Processing attack '{attack}' with {len(sub)} rows")

        relevant_cols = [
            c for c in all_group_cols
            if c == "attack" or not sub[c].isna().all()
        ]

        print(f"Computing ASR for attack '{attack}' with grouping columns: {relevant_cols}")

        # Then we group by the other relevant columns
        grouped = sub.groupby(relevant_cols, dropna=False)
        for key, g in grouped:
            n_runs = len(g)
            n_success = int(g["success"].sum())
            asr = n_success / n_runs if n_runs else float("nan")

            row = dict(zip(relevant_cols, key if isinstance(key, tuple) else (key,)))
            row["n_runs"] = n_runs
            row["n_success"] = n_success
            row["asr"] = asr
            if "clean_acc" in g.columns:
                row["clean_acc_mean"] = g["clean_acc"].mean()
            if "poisoned_acc" in g.columns:
                row["poisoned_acc_mean"] = g["poisoned_acc"].mean()
            results.append(row)

    summary = pd.DataFrame(results)
    summary = summary[summary["n_runs"] >= min_runs]
    summary = summary.sort_values(
        ["attack", "model", "asr"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return summary


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute ASR per model/attack/params.")

    # Adding arguments for output file and some filtering options
    parser.add_argument("csv_path", help="Path to results CSV")
    parser.add_argument("--out", default=None, help="Path to write summary CSV")
    parser.add_argument("--min-runs", type=int, default=1, help="Only report configs with at least this many runs (default: 1)")
    parser.add_argument("--attack", default=None, help="Filter to a single attack name before analysis")
    parser.add_argument("--top", type=int, default=None, help="Only print the top N configs by ASR (per attack)")
    args = parser.parse_args()

    # Extract arguments
    csv_path = args.csv_path
    min_runs = args.min_runs
    attack_filter = args.attack
    top_n = args.top
    out_file = args.out

    # read the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter to a specific attack if requested
    if args.attack:
        df = df[df["attack"] == attack_filter]
        if df.empty:
            sys.exit(f"No rows found for attack='{attack_filter}' in the CSV.")

    # Compute the ASR summary
    summary = compute_asr(df, min_runs=min_runs)

    # Print the summary to stdout
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Print the top N configs by ASR if requested, otherwise print all
    if top_n:
        printed = (summary.groupby("attack", group_keys=False).apply(lambda g: g.sort_values("asr", ascending=False).head(top_n)))
    else:
        printed = summary
    print(printed.to_string(index=False))

    # Quick overall rollup: ASR per (model, attack) ignoring finer params
    print("\n=== Overall ASR per model x attack (all param settings pooled) ===")
    overall = (df.groupby(["model", "attack"])["success"].agg(n_runs="count", n_success="sum").reset_index())
    overall["asr"] = overall["n_success"] / overall["n_runs"]
    overall["asr %"] = overall["asr"] * 100
    overall = overall.sort_values(["attack", "asr"], ascending=[True, False])

    print(overall.to_string(index=False))

    if out_file:
        summary.to_csv(out_file, index=False)
        print(f"\nFull per-config summary written to: {out_file}")

        overall_out_path = args.out.replace(".csv", "_overall.csv")

        overall.to_csv(overall_out_path, index=False)
        print(f"\nOverall summary written to: {overall_out_path}")


if __name__ == "__main__":
    main()