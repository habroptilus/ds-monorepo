class BasicTrainer:
    """普通の学習を行う."""

    def __init__(self):
        pass

    def run(self, train, valid, model):
        # 学習
        model.fit(train, valid)

        return {"model": model}

    def return_flag(self):
        return "basic"
