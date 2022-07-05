class JobsConfig:
    """実験設定yamlファイルを解析してSeedJobのパラメータのdictを作成する.

    各jobに対し、以下のルールで解決する
    nameキーを取り出す(指定がなければ自動でjobN(N番目のジョブ)が割り当てられる.)
    refキーがある場合、refキーをnameキーの下にそのままコピーして次のジョブへ
    refキーがない場合、paramsキーを取り出してsharedをjob固有設定で上書きしてnameキーの下にコピーして次のジョブへ
    refもparamsもないとwarningが出る.
    """

    def __init__(self, config):
        self.config = config

    def build(self):
        config = self.config.copy()
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

            config_dict[name]["params"] = copied_shared
        return config_dict


class StackingConfig:
    """実験設定yamlファイルを解析してStacking部分のパラメータを生成する."""

    def __init__(self, config) -> None:
        self.config = config

    def build(self):
        config = self.config.copy()
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
