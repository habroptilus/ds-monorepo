from lilac.core.factory_base import FactoryBase
from lilac.features.nlp.word_vectorizers.w2v_vectorizer import W2VVectorizer


class WordVectorizerFactory(FactoryBase):
    def __init__(self):
        str2model = {
            "w2v": W2VVectorizer,
        }
        super().__init__(str2model=str2model)
