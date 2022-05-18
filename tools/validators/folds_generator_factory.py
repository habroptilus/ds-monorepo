from tools.validators.folds_generators import (KFoldsGenerator,
                                               GroupKFoldsGenerator,
                                               StratifiedFoldsGenerator,
                                               StratifiedGroupKFoldGenerator)
from tools.core.factory_base import FactoryBase


class FoldsGeneratorFactory(FactoryBase):
    def __init__(self, custom_members=None):
        str2model = {
            "kfold": KFoldsGenerator,
            "stratified": StratifiedFoldsGenerator,
            "group": GroupKFoldsGenerator,
            "stratified_group": StratifiedGroupKFoldGenerator
        }
        super().__init__(str2model, custom_members)

    def run(self, model_str, fold_num, seed=None, params=None):
        return super().run(model_str, fold_num, seed, params)

    def get_params(self, fold_num, seed, params):
        if params is None:
            params = {}

        params["fold_num"] = fold_num
        params["seed"] = seed
        return params
