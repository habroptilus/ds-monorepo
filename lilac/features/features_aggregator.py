import pandas as pd


class FeaturesAggregator:
    """特徴量生成クラスを動かして出力をまとめる.pipelineのbuildも行う."""

    def __init__(self, target_col, factory, settings):
        """settingsの例:

        settings = [
            {"model_str": "count", "params":{
                "input_cols": ["neighbourhood", "room_type"]}},
            {"model_str": "onehot","params":{
                "input_cols": ["neighbourhood", "room_type"]}},
            {"model_str": "ordinal","params":{"input_cols": ["neighbourhood", "room_type"]},
            {
                "model_str": "pipeline",
                "params": {
                    "use_previous_cols": True, or [True]
                    "feature_generators":[
                        {"model_str": "bert", "params": {"input_col": "name",  "max_len":128}},
                        {"model_str": "pca", "params": {"n_components":20,"random_state":42,"col_mark":"bert"}},
                    ]
                }
            }
        ]
        """
        self.target_col = target_col
        self.factory = factory
        self.generators = self.build_generators(settings)

    def build_generators(self, settings):
        generators = []
        for setting in settings:
            if setting["model_str"] == "pipeline":
                # buildしてparamsを書き換える
                model_str = "pipeline"
                feature_generators = self.build_generators(setting["params"]["feature_generators"])
                params = setting.get("params", {}).copy()
                params["feature_generators"] = feature_generators
            else:
                model_str = setting["model_str"]
                params = setting.get("params")
            generator = self.factory.run(model_str, params)
            generators.append(generator)
        return generators

    def run(self, train_data, test_data):
        y = train_data[self.target_col]
        train_data = train_data.drop(self.target_col, axis=1)
        train_result = train_data.copy()
        test_result = test_data.copy()
        features_n = len(self.generators)
        for i, generator in enumerate(self.generators):
            print(f"[{i+1}/{features_n}] {generator.__class__.__name__}")
            train, test = generator.run(train_data, test_data)
            dup_train = set(train_result.columns) & set(train.columns)
            dup_test = set(test_result.columns) & set(test.columns)
            if dup_train or dup_test:
                raise Exception(f"Aggregation error. Duplicated cols are about to be added.{dup_train,dup_test}")
            train_result = pd.concat([train_result, train], axis=1)
            test_result = pd.concat([test_result, test], axis=1)
        train_result[self.target_col] = y
        return train_result, test_result
