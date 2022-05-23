from lilac.director.experiment_director import ExperimentDirector
import fire
import yaml
from pathlib import Path


def run(project_name, filename, data_dir="data", input_dir="config", output_dir="output"):
    """runコマンドの実装.

    :project_name: プロジェクト名
    :filepath: プロジェクト以下のyamlファイルパス
    :output_path: 
    """
    config_path = Path(
        f"projects/{project_name}/{input_dir}/{filename}")
    output_path = f"projects/{project_name}/{data_dir}/{output_dir}/{config_path.stem}.json"

    with config_path.open("r") as yml:
        experiment_config = yaml.safe_load(yml)

    ExperimentDirector(
        output_path=output_path).run(experiment_config)


def main():
    fire.Fire({"run": run})
