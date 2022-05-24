import glob
from pathlib import Path
import json
output_path_list = glob.glob("data/output/*.json")


for output_path in sorted(output_path_list):
    p = Path(output_path)
    with p.open("r") as f:
        data = json.load(f)
        score = data["output"].get("score")
        title = data.get("meta", {}).get("title")
        print(f"{p.stem}: {score}   {title}")
