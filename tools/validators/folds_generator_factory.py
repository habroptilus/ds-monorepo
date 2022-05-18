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
