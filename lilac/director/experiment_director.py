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
            raise Exception(
                "'stacking_settings' key was Not found in 'stacking'.")
        return {
            "stacking_settings": stacking_settings,
            **shared
        }


class ExperimentDirector:
    """configファイルを読み込んだdictを受け取り実験を行って結果を返す."""

    def run(self, config):
        jobs_config = JobsConfigResolver().run(config)
        output_list = []
        job_factory = JobFactory()

        for name, params in jobs_config.items():
            print(f"Job '{name}' is running...")
            # これで不要なパラメータが入っていても取り除いてくれる
            # Factory経由にしないと不要なパラメータが入っていたらエラーになる
            basic_seed_job = job_factory.run(
                model_str="basic_seed", params=params)
            output = basic_seed_job.run()
            output_list.append(output)

        stacking_config = StackingConfigResolver().run(config)

        if stacking_config is not None:
            # これで不要なパラメータが入っていても取り除いてくれる
            # Factory経由にしないと不要なパラメータが入っていたらエラーになる
            stacking_job = job_factory.run(
                model_str="stacking", params=stacking_config)
            stacking_output = stacking_job.run(output_list)

        return {
            "seed_jobs": output_list,
            "stacking": stacking_output
        }
