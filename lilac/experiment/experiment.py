import json

from lilac.jobs.job_factory import JobFactory


class Experiment:
    """複数のseed jobとstacking jobをまとめたもの."""

    def __init__(self) -> None:
        self.job_factory = JobFactory()

    def run(self, jobs_config, stacking_config):
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
            elif "params" in job_settings:
                # これで不要なパラメータが入っていても取り除いてくれる
                # Factory経由にしないと不要なパラメータが入っていたらエラーになる
                print(f"Generating output by running job '{name}'...")
                basic_seed_job = self.job_factory.run(
                    model_str="basic_seed", params=job_settings["params"], allow_extra_params=False
                )
                output = basic_seed_job.run()
            else:
                raise Exception("Neigher 'ref' nor 'params' are set.")
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
