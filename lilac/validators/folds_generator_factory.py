from lilac.core.factory_base import FactoryBase
from lilac.validators.folds_generators import (
    GroupKFoldsGenerator,
    KFoldsGenerator,
    StratifiedFoldsGenerator,
    StratifiedGroupKFoldGenerator,
)


class FoldsGeneratorFactory(FactoryBase):
    def __init__(self, register_from=None):
        str2model = {
            "kfold": KFoldsGenerator,
            "stratified": StratifiedFoldsGenerator,
            "group": GroupKFoldsGenerator,
            "stratified_group": StratifiedGroupKFoldGenerator,
        }
        super().__init__(str2model, register_from)
