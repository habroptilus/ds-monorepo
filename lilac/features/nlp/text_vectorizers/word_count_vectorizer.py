from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lilac.features.nlp.text_vectorizers.text_vectorizer_base import TextVectorizerBase


class WordCountVectorizer(TextVectorizerBase):
    str2vectorizer = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}

    def __init__(self, vectorizer_str):
        self.vectorizer_str = vectorizer_str
        vectorizer = self.str2vectorizer.get(vectorizer_str)
        if vectorizer is None:
            raise Exception(f"Invalid vectorizer str : '{vectorizer_str}'")
        self.vectorizer = vectorizer()

    def fit(self, docs):
        self.vectorizer.fit(docs)
        return self

    def transform(self, docs):
        vectors = self.vectorizer.transform(docs)
        # transform sparse matrix to numpy array.
        return vectors.toarray()

    def return_flag(self):
        return f"{self.vectorizer_str}"
