import glob
import json
from pathlib import Path

import fire
import yaml


def run(project_name, filename, data_dir="data", input_dir="config", output_dir="output"):
    """runコマンドの実装.

    :project_name: プロジェクト名
    :filepath: プロジェクト以下のyamlファイルパス
    :output_path:
    """
    from lilac.director.experiment_director import ExperimentDirector

    config_path = Path(f"projects/{project_name}/{input_dir}/{filename}")
    output_path = f"projects/{project_name}/{data_dir}/{output_dir}/{config_path.stem}.json"

    with config_path.open("r") as yml:
        experiment_config = yaml.safe_load(yml)

    ExperimentDirector(output_path=output_path).run(experiment_config)


def result_list(project_name, num=5, data_dir="data", output_dir="output"):
    output_path_list = glob.glob(f"projects/{project_name}/{data_dir}/{output_dir}/*.json")

    for output_path in list(reversed(sorted(output_path_list)))[:num]:
        p = Path(output_path)
        with p.open("r") as f:
            data = json.load(f)
            score = data["output"].get("score")
            title = data.get("meta", {}).get("title")
            print(f"{p.stem}: {score}   {title}")


def result_detail(project_name, experiment_id, data_dir="data", output_dir="output"):
    output_path = f"projects/{project_name}/{data_dir}/{output_dir}/{experiment_id}.json"
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
        print("[Result]")
        print(f"CV: {score}")


def main():
    fire.Fire({"run": run, "list": result_list, "detail": result_detail})
