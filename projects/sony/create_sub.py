import argparse
import json

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="create sub")

    parser.add_argument("exp")
    args = parser.parse_args()

    with open(f"data/output/{args.exp}.json", "r") as f:
        result = json.load(f)

    pred = result["output"]["pred"]

    sub = pd.read_csv("data/submit_sample.csv", names=["id", "pred"])
    sub["pred"] = pred

    sub.to_csv(f"data/subs/{args.exp}.csv", header=False, index=False)
