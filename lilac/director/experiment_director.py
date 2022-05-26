import json
from pathlib import Path

from lilac.jobs.job_factory import JobFactory


class JobsConfigResolver:
    """実験設定yamlファイルを解析してSeedJobのパラメータのdictを作成する.

    keyはnameを指定していればname, していなければ自動でjobN(N番目のジョブ)が割り当てられる.
    valueはsharedを各jobのparamsで置き換えたもの.
    paramsがないとwarningが出る.
    """

    def run(self, config):
        config = config.copy()
        shared = config.get("shared", {})
        jobs = config.get("jobs")
        if len(jobs) == 0:
            raise Exception("'jobs' key should be a list that has one job at least.")
        config_dict = {}
        for i, job in enumerate(jobs):
            name = job.get("name", f"job{i+1}")

            copied_shared = shared.copy()
            params = job.get("params")
            if params is None:
                print(f"[WARNING] job '{name}' is using default parameters. Are you sure?")
            else:
                copied_shared.update(params)
            config_dict[name] = copied_shared
        return config_dict


class StackingConfigResolver:
    """実験設定yamlファイルを解析してStacking部分のパラメータを生成する."""

    def run(self, config):
        config = config.copy()
        shared = config.get("shared", {})
        stacking = config.get("stacking")
        if stacking is None:
            return None

        stacking_shared = stacking.pop("shared", {})
        shared.update(stacking_shared)
        shared
        stacking_settings = stacking.get("stacking_settings")
        if stacking_settings is None:
            raise Exception("'stacking_settings' key was Not found in 'stacking'.")
        return {"stacking_settings": stacking_settings, **shared}


class ExperimentDirector:
    """configファイルを読み込んだdictを受け取り実験を行って返す.パスが指定されていたらJsonにdumpする.metaキーがあればそれをoutputにいれる"""

    def __init__(self, output_path=None):
        self.output_path = output_path
        self.job_factory = JobFactory()

    def run(self, config):
        jobs_config = JobsConfigResolver().run(config)
        stacking_config = StackingConfigResolver().run(config)

        result = {}

        output_dict = self.run_seed_jobs(jobs_config)

        result["seed_jobs"] = output_dict

        if stacking_config is not None:
            stacking_output = self.run_stacking_job(stacking_config, output_dict)
            result["stacking"] = stacking_output

        final_output = self.get_final_output(result)
        if final_output:
            result["output"] = final_output
            print(f"CV: {final_output['score']}")

        if "meta" in config:
            result["meta"] = config["meta"]
        self.dump_result(result)

        return result

    def run_stacking_job(self, stacking_config, output_dict):
        # これで不要なパラメータが入っていても取り除いてくれる
        # Factory経由にしないと不要なパラメータが入っていたらエラーになる
        stacking_job = self.job_factory.run(model_str="stacking", params=stacking_config)
        return stacking_job.run(output_dict.values())

    def run_seed_jobs(self, jobs_config):
        output_dict = {}
        for name, params in jobs_config.items():
            print(f"Job '{name}' is running...")
            # これで不要なパラメータが入っていても取り除いてくれる
            # Factory経由にしないと不要なパラメータが入っていたらエラーになる
            basic_seed_job = self.job_factory.run(model_str="basic_seed", params=params)
            output = basic_seed_job.run()
            output_dict[name] = output
        return output_dict

    def get_final_output(self, result):
        """実験全体で最終的な出力となるoutputを返す.

        :seed jobが一つの場合 : そのSeedJobのoutput
        :seed jobが複数かつstackingを実施した場合 : stackingの最後の層のoutput
        :seed jobが複数なのにstackingがない場合 : Warningを出してNoneを返す.
        """
        if len(result["seed_jobs"]) == 1:
            return list(result["seed_jobs"].values())[0]
        elif "stacking" in result:
            return result["stacking"][-1][0]
        else:
            print(
                "[WARNING] There are multiple SeedJobs but No stacking was conducted. Final result couldn't be specified."
            )

    def dump_result(self, result):
        with Path(self.output_path).open("w") as f:
            json.dump(result, f)
