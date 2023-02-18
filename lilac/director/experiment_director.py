import json

from lilac.experiment.config import JobsConfig, StackingConfig
from lilac.experiment.experiment import Experiment


class ProjectPathProcessor:
    """model_dirやfeatures_dir, train_pathなどを設定に追加する."""

    def __init__(self, project_dir):
        self.project_dir = project_dir

    def run(self, jobs_config, stacking_config, model_dir):
        jobs_config = self.set_jobs_config(jobs_config=jobs_config, model_dir=model_dir)
        if stacking_config:
            stacking_config = self.set_stakcing_config(stacking_config=stacking_config)
        return jobs_config, stacking_config

    def set_jobs_config(self, jobs_config, model_dir=None):
        for name, job in jobs_config.items():
            if "ref" in job:
                continue

            if "params" in job:
                update_dict = self.create_update_dict(job["params"])
                if model_dir:
                    update_dict["model_dir"] = f"{model_dir}/{name}"
                job["params"].update(update_dict)
        return jobs_config

    def set_stakcing_config(self, stacking_config):
        """model_dirはstackingは未対応なので設定しない.TODO."""
        stacking_config.update(self.create_update_dict(stacking_config))
        return stacking_config

    def create_update_dict(self, params):
        """TODO:register_fromもproject名から作るように追加する."""
        return {key: self.project_dir / params[key] for key in ["features_dir", "train_path", "test_path"]}


class ExperimentCliDirector:
    """cliから呼び出される.

    - Configのbuild
    - project名からpathを生成して追加
    - Experimentに設定を渡して実行
    - metaデータを追加しjsonとして保存
    """

    def __init__(
        self,
        project_dir,
        output_filename,
        output_dir="output",
        data_dir="data",
        model_dir=None,
    ):
        self.output_path = project_dir / data_dir / output_dir / output_filename
        self.model_dir = project_dir / data_dir / model_dir if model_dir else None
        self.project_dir = project_dir

    def run(self, config):
        """
        config: yamlを読み込んだ物.dict形式
        model_dir: save_model=Trueの時はmodelディレクトリ、FalseのときはNone
        """

        # yamlファイルからsharedとかの処理をする
        jobs_config = JobsConfig(config).build()
        stacking_config = StackingConfig(config).build()

        # model_dirやfeatures_dirなどのpathを変換する.
        processor = ProjectPathProcessor(project_dir=self.project_dir)
        jobs_config, stacking_config = processor.run(
            jobs_config=jobs_config, stacking_config=stacking_config, model_dir=self.model_dir
        )

        # Experimentに設定を渡す
        result = Experiment().run(jobs_config=jobs_config, stacking_config=stacking_config)

        # メタデータを追加
        if "meta" in config:
            result["meta"] = config["meta"]

        # jsonとして保存
        self.dump_result(result)

        return result

    def dump_result(self, result):
        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True)
        with self.output_path.open("w") as f:
            json.dump(result, f)
