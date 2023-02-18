class TextVectorizerBase:
    """text vectorizerの基底クラス."""

    def __init__(self):
        raise Exception("Not implemented.")

    def transform(self, docs):
        raise Exception("Not implemented.")

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def fit(self, docs):
        raise Exception("Not implemented.")
