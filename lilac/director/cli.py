import glob
import json
from pathlib import Path

import fire
import pandas as pd
import yaml


def run(
    project_name,
    filename,
    data_dir="data",
    input_dir="config",
    output_dir="output",
    model_dir="models",
    save_model=False,
):
    """runコマンドの実装.

    :project_name: プロジェクト名
    :filepath: プロジェクト以下のyamlファイルパス
    :input_dir: 実験設定ファイルyaml置き場
    :data_dir: 生成物置き場。outputやmodel,featuresが出力されるところ
    :output_dir: json出力先ディレクトリ
    :model_dir: model出力先ディレクトリ
    :save_model: modelを保存するか.
    """
    from lilac.director.experiment_director import ExperimentCliDirector

    project_dir = Path(f"projects/{project_name}")
    config_path = project_dir / f"{input_dir}/{filename}"

    with config_path.open("r") as yml:
        experiment_config = yaml.safe_load(yml)

    model_dir = Path(f"{model_dir}/{config_path.stem}") if save_model else None
    output_filename = f"{config_path.stem}.json"

    ExperimentCliDirector(
        project_dir=project_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        output_filename=output_filename,
    ).run(experiment_config)


def result_list(project_name, num=5, data_dir="data", output_dir="output"):
    output_path_list = glob.glob(f"projects/{project_name}/{data_dir}/{output_dir}/*.json")
    num = min(len(output_path_list), num)
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


def plot_importance(project_name, experiment_id, job_name="job1", data_dir="data", output_dir="output", num=20):
    from lilac.core.utils import plot_feature_importance

    output_path = f"projects/{project_name}/{data_dir}/{output_dir}/{experiment_id}.json"
    p = Path(output_path)
    with p.open("r") as f:
        data = json.load(f)
        job_result = data["seed_jobs"].get(job_name)

        if job_result is None:
            raise Exception(
                f"{job_name} is not in {experiment_id}. Please add '-j <job_name>'. You can use: {list(data['seed_jobs'].keys())}"
            )
        additional = job_result.get("additional")
        if additional is None or len(additional) == 0 or additional[0] is None:
            raise Exception(f"{job_name} in {experiment_id} doesn't have feature importance.")
        importance = additional[0].get("importance")
        if importance is None:
            raise Exception(f"{job_name} in {experiment_id} doesn't have feature importance.")

        plot_feature_importance(pd.DataFrame(importance), max_n=num)


def init_project(project_name, data_dir="data", config_dir="config", output_dir="output"):
    project_dir = Path(f"projects/{project_name}")
    for dir in [data_dir, config_dir, output_dir]:
        target_dir = project_dir / dir
        target_dir.mkdir()


def main():
    fire.Fire(
        {"run": run, "list": result_list, "detail": result_detail, "plot": plot_importance, "init": init_project}
    )
