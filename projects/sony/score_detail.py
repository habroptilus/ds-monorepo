import json
import sys
from pathlib import Path

experiment_id = sys.argv[1]

output_path = f"data/output/{experiment_id}.json"
p = Path(output_path)
with p.open("r") as f:
    data = json.load(f)
    title = data.get("meta", {}).get("title")
    print(f"Title: {title}")
    for i, (name, job_result) in enumerate(data["seed_jobs"].items()):
        print(f"[{i+1}/{len(data['seed_jobs'])}]")
        score = job_result["score"]
        print(f"Name: {name}")
        print(f"CV: {score}")
    score = data["output"].get("score")
    print(f"[Result]")
    print(f"CV: {score}")
