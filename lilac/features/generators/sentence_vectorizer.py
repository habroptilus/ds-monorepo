import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from lilac.features.generator_base import FeaturesBase
from lilac.features.generators.decomposer_features import DecompositionFeatures, StandardizedDecomposer
from lilac.features.wrappers.features_pipeline import FeaturesPipeline


class WordCountVectorizer(FeaturesBase):
    """BoWやTfidfで文書をベクトル化する.次元圧縮は行わない.
    csr_martix->ndarrayに変換してdataframeにして保存するので大きいと辛いかも.
    """

    def __init__(self, input_col, vectorizer_str, features_dir=None):
        self.input_col = input_col
        self.vectorizer_str = vectorizer_str
        super().__init__(features_dir)

    def fit(self, df):
        str2vectorizer = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}
        vectorizer = str2vectorizer.get(self.vectorizer_str)
        if vectorizer is None:
            raise Exception(f"Invalid vectorizer str : '{ self.vectorizer_str}'")

        self.vectorizer = vectorizer().fit(df[self.input_col])
        return self

    def transform(self, df):
        vecs = self.vectorizer.transform(df[self.input_col])
        vecs = vecs.toarray()
        return pd.DataFrame(
            vecs,
            columns=[f"{self.input_col}_{self.vectorizer_str}-{i+1}" for i in range(vecs.shape[1])],
        )


class BertVectorizer(FeaturesBase):
    """Bertの学習済みモデルを使って768次元のベクトルを出力する."""

    def __init__(self, input_col, model_name="bert-base-uncased", max_len=128, features_dir=None):
        self.input_col = input_col
        self.max_len = max_len
        self.model_name = model_name
        super().__init__(features_dir)

    def fit(self, df):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        return self

    def transform(self, df):
        docs = df[self.input_col]
        result = []
        for doc in tqdm(docs):
            inp = self.tokenizer.encode(doc)
            len_inp = len(inp)

            if len_inp >= self.max_len:
                inputs = inp[: self.max_len]
                masks = [1] * self.max_len
            else:
                inputs = inp + [0] * (self.max_len - len_inp)
                masks = [1] * len_inp + [0] * (self.max_len - len_inp)

            inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
            masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

            outputs = self.bert_model(inputs_tensor, masks_tensor)
            seq_out = outputs.last_hidden_state

            if torch.cuda.is_available():
                # 0番目は [CLS] token, 768 dim の文章特徴量
                vector = seq_out[0][0].cpu().detach().numpy()
            else:
                vector = seq_out[0][0].detach().numpy()
            result.append(vector)
        cols = [f"bert_{self.input_col}_{i+1}" for i in range(768)]
        return pd.DataFrame(np.array(result), columns=cols)


class DecomposedSentenceVectoizer(FeaturesPipeline):
    """BertやWordCountのベクトルをPCAやUMAP,NMF,SVDなどで次元削減する.次元削減前の標準化するかどうかも選べる."""

    def __init__(
        self,
        input_col,
        vectorizer_str,
        decomposer_str,
        n_components,
        model_name="bert-base-uncased",
        max_len=128,
        standardize=False,
        seed=None,
        features_dir=None,
    ):
        if vectorizer_str == "bert":
            vectorizer = BertVectorizer(
                input_col=input_col, model_name=model_name, max_len=max_len, features_dir=features_dir
            )
        elif vectorizer_str in ["bow", "tfidf"]:
            vectorizer = WordCountVectorizer(
                input_col=input_col, vectorizer_str=vectorizer_str, features_dir=features_dir
            )
        else:
            raise Exception(f"Invalid vectorizer_str '{vectorizer_str}'")

        if standardize:
            decomposer = StandardizedDecomposer
        else:
            decomposer = DecompositionFeatures

        # TODO : ここ本当は一意にしたいけど長くなってしまう...
        prefix = f"{input_col}_{vectorizer_str}_{decomposer_str}_{n_components}"

        decomposer = decomposer(
            decomposer_str=decomposer_str,
            n_components=n_components,
            seed=seed,
            features_dir=features_dir,
            prefix=prefix,
        )

        super().__init__(feature_generators=[vectorizer, decomposer], use_prev_only=True, features_dir=features_dir)
