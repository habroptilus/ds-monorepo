from gensim.models.word2vec import Word2Vec

from lilac.features.nlp.word_vectorizers.word_vectorizer_base import WordVectorizerBase


class W2VVectorizer(WordVectorizerBase):
    def __init__(self, vector_size, seed=None, sep=" "):
        self.vector_size = vector_size
        self.seed = seed
        self.sep = sep

    def fit(self, docs):
        sentences = [doc.split(self.sep) for doc in docs]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, seed=self.seed)

    def transform(self, word: str):
        """未知語の場合はNoneを返す.(オプションで似ている単語を返すでもいいが...)"""
        if word not in self.model.wv:
            return None
        return self.model.wv[word]

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)

    def get_vectors(self):
        return self.model.wv.vectors

    def get_words(self):
        return self.model.wv.index_to_key
