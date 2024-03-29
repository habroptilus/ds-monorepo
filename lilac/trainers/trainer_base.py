from abc import ABCMeta, abstractmethod


class TrainerBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, train, valid, model_factory, model_params):
        pass

    @abstractmethod
    def return_flag(self):
        pass
