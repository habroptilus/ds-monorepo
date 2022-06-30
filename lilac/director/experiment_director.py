import json
from pathlib import Path

from lilac.jobs.job_factory import JobFactory


class JobsConfigResolver:
    """実験設定yamlファイルを解析してSeedJobのパラメータのdictを作成する.

    各jobに対し、以下のルールで解決する
    nameキーを取り出す(指定がなければ自動でjobN(N番目のジョブ)が割り当てられる.)
    refキーがある場合、refキーをnameキーの下にそのままコピーして次のジョブへ
    refキーがない場合、paramsキーを取り出してsharedをjob固有設定で上書きしてnameキーの下にコピーして次のジョブへ
    refもparamsもないとwarningが出る.
    """

    def run(self, config, model_dir):
        config = config.copy()
        shared = config.get("shared", {})
        jobs = config.get("jobs")
        if len(jobs) == 0:
            raise Exception("'jobs' key should be a list that has one job at least.")
        config_dict = {}
        names = set()
        for i, job in enumerate(jobs):
            name = job.get("name", f"job{i+1}")
            if name in names:
                raise Exception(f"Duplicated job name found:'{name}'.")
            names.add(name)
            config_dict[name] = {}
            ref = job.get("ref")
            if ref is not None:
                config_dict[name]["ref"] = ref
                continue

            copied_shared = shared.copy()

            params = job.get("params")
            if params is None:
                print(f"[WARNING] In Job '{name}', neither params nor ref are set. Are you sure?")
            else:
                copied_shared.update(params)

            if model_dir:
                # model_dirが設定されているときはjobごとに違うmodel_dirを設定する.
                copied_shared["model_dir"] = f"{model_dir}/{name}"
            config_dict[name]["params"] = copied_shared
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
        stacking_settings = stacking.get("stacking_settings")
        if stacking_settings is None:
            raise Exception("'stacking_settings' key was Not found in 'stacking'.")
        return {"stacking_settings": stacking_settings, **shared}


class ExperimentDirector:
    """configファイルを読み込んだdictを受け取り実験を行って返す.パスが指定されていたらJsonにdumpする.metaキーがあればそれをoutputにいれる"""

    def __init__(self, output_path=None):
        self.output_path = Path(output_path)
        self.job_factory = JobFactory()

    def run(self, config, model_dir):
        """
        config: yamlを読み込んだ物.dict形式
        model_dir: save_model=Trueの時はmodelディレクトリ、FalseのときはNone
        """
        jobs_config = JobsConfigResolver().run(config, model_dir)
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
        """seed jobに対応する結果ファイルを取得する.

        * refがあれば結果ファイルを読み込む
            * jobに指定があればそのjobを、なければoutputを読み込む.
        * refがなければparamsを使ってjobを作成しrunして結果を得る.
        """
        output_dict = {}
        for name, job_settings in jobs_config.items():
            print(f"Job '{name}' is running...")
            ref = job_settings.get("ref")
            if ref:
                src_file = self.output_path.parent / ref["src"]
                print(f"Loading output from '{src_file}'...")
                with src_file.open("r") as f:
                    result = json.load(f)
                    job_name = ref.get("job")
                    if job_name is None:
                        output = result["output"]
                    else:
                        output = result["seed_jobs"].get(job_name)
                        if output is None:
                            raise Exception(f"Job name '{job_name}' is not found in Experiment '{ref['src']}'.")
                print(f"CV: {output['score']}")
            else:
                # これで不要なパラメータが入っていても取り除いてくれる
                # Factory経由にしないと不要なパラメータが入っていたらエラーになる
                print(f"Generating output by running job '{name}'...")
                basic_seed_job = self.job_factory.run(
                    model_str="basic_seed", params=job_settings["params"], allow_extra_params=False
                )
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
            print("[WARNING] There are multiple SeedJobs but No stacking was conducted.")

    def dump_result(self, result):
        with self.output_path.open("w") as f:
            json.dump(result, f)
