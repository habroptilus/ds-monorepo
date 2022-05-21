from tools.trainers.trainer_base import TrainerBase


class BasicTrainer(TrainerBase):
    """普通の学習を行う."""

    def run(self, train, valid, model):
        # 学習
        model.fit(train, valid)

        return model

    def return_flag(self):
        return "basic"
