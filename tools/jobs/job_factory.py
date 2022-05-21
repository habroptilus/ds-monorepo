from tools.core.factory_base import FactoryBase
from tools.jobs.job_base import BasicSeedJob, StackingJob


class JobFactory(FactoryBase):
    """JobクラスのFactory."""

    def __init__(self, register_from=None):
        str2model = {
            "basic_seed": BasicSeedJob,
            "stacking": StackingJob
        }
        super().__init__(str2model, register_from)
