from lilac.director.experiment_director import ExperimentDirector
import fire
import yaml


def run_experiment(filepath):
    with open(filepath, 'r') as yml:
        experiment_config = yaml.safe_load(yml)
    result = ExperimentDirector().run(experiment_config)
    score = result["stacking"][-1][0]["score"]
    print(f"CV : {score}")


def main():
    fire.Fire({"run": run_experiment})
