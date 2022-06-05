import argparse
import json

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="create sub")

    parser.add_argument("exp")
    args = parser.parse_args()

    with open(f"data/output/{args.exp}.json", "r") as f:
        result = json.load(f)

    submit_df = pd.DataFrame({"y": result["output"]["pred"]})
    submit_df.index.name = "id"
    submit_df.to_csv(f"data/subs/{args.exp}.csv")
