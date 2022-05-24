from lilac.trainers.trainer_base import TrainerBase


class BasicTrainer(TrainerBase):
    """普通の学習を行う."""

    def run(self, train, valid, model_factory, model_params):
        model = model_factory.run(**model_params)
        # 学習
        model.fit(train, valid)

        return model

    def return_flag(self):
        return "basic"
