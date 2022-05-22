import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from lilac.features.generator_base import FeaturesBase
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF


class WordCountVectorizer(FeaturesBase):
    """BoWやTfidfで文書をベクトル化し、SVD等の次元圧縮をする.(PCAはスパースに対応していないらしい)"""
    str2vectorizer = {
        "bow": CountVectorizer,
        "tfidf": TfidfVectorizer
    }
    str2decomposer = {
        "svd": TruncatedSVD,
        "nmf": NMF
    }

    def __init__(self, input_col, vectorizer_str, decomposer_str,  n_components, random_state=None, features_dir=None):
        self.input_col = input_col
        self.vectorizer_str = vectorizer_str
        self.decomposer_str = decomposer_str
        self.random_state = random_state
        self.n_components = n_components
        super().__init__(features_dir)

    def fit(self, df):
        if self.vectorizer_str not in self.str2vectorizer:
            raise Exception(
                f"Invalid vectorizer str : '{ self.vectorizer_str}'")
        self.vectorizer = self.str2vectorizer[self.vectorizer_str]()

        if self.decomposer_str not in self.str2decomposer:
            raise Exception(
                f"Invalid decomposer_str : '{ self.decomposer_str}'")
        self.decomposer = self.str2decomposer[self.decomposer_str](
            n_components=self.n_components, random_state=self.random_state)
        vecs = self.vectorizer.fit_transform(df[self.input_col])
        self.decomposer.fit(vecs)
        return self

    def transform(self, df):
        vecs = self.vectorizer.transform(df[self.input_col])
        vecs = self.decomposer.transform(vecs)
        return pd.DataFrame(
            vecs, columns=[f"{self.input_col}_{self.vectorizer_str}_{self.decomposer_str}_{i+1}" for i in range(self.n_components)])


class BertVectorizer(FeaturesBase):
    """Bertの学習済みモデルを使って768次元のベクトルを出力する."""

    def __init__(self, input_col, model_name='bert-base-uncased', max_len=128, features_dir=None):
        self.input_col = input_col
        self.max_len = max_len
        self.model_name = model_name
        super().__init__(features_dir)

    def fit(self, df):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        return self

    def transform(self, df):
        docs = df[self.input_col]
        result = []
        for doc in tqdm(docs):
            inp = self.tokenizer.encode(doc)
            len_inp = len(inp)

            if len_inp >= self.max_len:
                inputs = inp[:self.max_len]
                masks = [1] * self.max_len
            else:
                inputs = inp + [0] * (self.max_len - len_inp)
                masks = [1] * len_inp + [0] * (self.max_len - len_inp)

            inputs_tensor = torch.tensor(
                [inputs], dtype=torch.long).to(self.device)
            masks_tensor = torch.tensor(
                [masks], dtype=torch.long).to(self.device)

            outputs = self.bert_model(inputs_tensor, masks_tensor)
            seq_out = outputs.last_hidden_state

            if torch.cuda.is_available():
                # 0番目は [CLS] token, 768 dim の文章特徴量
                vector = seq_out[0][0].cpu().detach().numpy()
            else:
                vector = seq_out[0][0].detach().numpy()
            result.append(vector)
        cols = [f"bert_{self.input_col}_{i+1}"for i in range(768)]
        return pd.DataFrame(np.array(result), columns=cols)
