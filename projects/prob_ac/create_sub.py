import argparse
import json

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="create sub")

    parser.add_argument("exp")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(f"projects/prob_ac/data/output/{args.exp}.json", "r") as f:
        result = json.load(f)

    raw_pred = result["output"]["raw_pred"]
    pred = [1 if value >= args.threshold else 0 for value in raw_pred]

    submit_df = pd.DataFrame({"y": pred})

    print(submit_df["y"].value_counts())
    submit_df.index.name = "id"
    submit_df.to_csv(f"projects/prob_ac/data/subs/{args.exp}.csv")
