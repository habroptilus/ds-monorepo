from lilac.core.factory_base import FactoryBase
from lilac.features.nlp.text_vectorizers.word_vector_based_vectorizer import WordVectorBasedVectorizer


class TextVectorizerFactory(FactoryBase):
    def __init__(self):
        str2model = {
            "word_based_text_vec": WordVectorBasedVectorizer,
        }
        super().__init__(str2model=str2model)
