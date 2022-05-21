import yaml
from tools.jobs.job_base import BasicSeedJob


class JobsConfigResolver:
    """実験設定yamlファイルを解析してSeedJobのパラメータのdictを作成する.

    keyはnameを指定していればname, していなければ自動でjobN(N番目のジョブ)が割り当てられる.
    valueはsharedを各jobのparamsで置き換えたもの.
    paramsがないとwarningが出る.
    """

    def run(self, config):
        shared = config.get("shared", {})
        jobs = config.get("jobs")
        if len(jobs) == 0:
            raise Exception(
                "'jobs' key should be a list that has one job at least.")
        config_dict = {}
        for i, job in enumerate(jobs):
            name = job.get("name", f'job{i+1}')

            copied_shared = shared.copy()
            params = job.get("params")
            if params is None:
                print(
                    f"[WARNING] job '{name}' is using default parameters. Are you sure?")
            else:
                copied_shared.update(params)
            config_dict[name] = copied_shared
        return config_dict


with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)


config = JobsConfigResolver().run(config)

output = BasicSeedJob(**config["seed43"]).run()

print(output["score"])
