from abc import ABCMeta, abstractmethod


class WordVectorizerBase(metaclass=ABCMeta):
    """word vectorizerの基底クラス."""

    def __init__(self):
        raise Exception("Not implemented.")

    @abstractmethod
    def transform(self, word):
        raise Exception("Not implemented.")

    @abstractmethod
    def fit(self, docs):
        raise Exception("Not implemented.")
