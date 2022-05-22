from lilac.ensemble.ensemble_runner_factory import EnsembleRunnerFactory


class StackingRunner:
    """1段目の出力を受け取り,複数のアンサンブルをまとめて実行する(stacking)."""

    def __init__(self, stacking_settings, shared):
        """
        :shared_params: ensemble runner生成に渡す共通のパラメータ.
        :settings : どのモデルを使って何層stackingするかの設定. List[str] or List[dict]
        ex)
        shared_params = {
            "target_col": target_col,
            "unused_cols": unused_cols,
            "folds_gen_settings": folds_gen_settings,
            "trainer_params": trainer_params,
            "use_original_cols": False
        }

        stacking_settings=[
            ["avg_rmsle","linear_rmsle","ridge_rmsle","lgbm_rmsle"],
            ["avg_rmsle"]
        ]
        or
        stacking_settings=[
            [
                {"model": "avg_rmsle", "params": params},
                {"model": "linear_rmsle","params": params},
                {"model": "ridge_rmsle", "params": params},
                {"model": "lgbm_rmsle", "params": params}
            ],
            [
                {"model": "avg_rmsle", "params": params}
            ]
        ]
        基本的にはshared_paramsがデフォルトで設定され、個別のensemble_runnerに設定を渡した時にはそのdictで上書きされる.
        出力はsettingsと同じshape
        """
        self.settings = stacking_settings
        self.shared_params = shared
        self.factory = EnsembleRunnerFactory()

    def run(self, output_list, train, test):
        result_list = []
        input_list = output_list
        for i, layer in enumerate(self.settings):
            print(f"Layer {i+1}")
            layer_results = []
            if type(layer) is not list:
                raise Exception(
                    f"Invalid type of layer: {type(layer), layer}. The type should be list.")
            for ensemble_item in layer:
                if isinstance(ensemble_item, dict):
                    ensemble_flag = ensemble_item["model"]
                    ensemble_params = ensemble_item["params"]
                    params = self.shared_params.copy()
                    # shared_paramsを上書き
                    params.update(ensemble_params)
                elif isinstance(ensemble_item, str):
                    ensemble_flag = ensemble_item
                    params = self.shared_params
                else:
                    raise Exception(
                        f"Type of ensemble_item should be dict or str: {type(ensemble_item)}")

                runner = self.factory.run(
                    model_str=ensemble_flag, params=params)

                result = runner.run(input_list, train, test)
                layer_results.append(result)
            result_list.append(layer_results)
            input_list = layer_results
        return result_list
