import numpy as np

from lilac.features.generator_base import FeaturesBase
from projects.prob_ac.custom.preprocess import flatten_keywords, preprocess_keywords


def apply_mapping_to_keywords(keyword, target_mapping):
    keywords = keyword.split(", ")
    return np.mean([target_mapping[word] for word in keywords])


class KeywordsLengthFeature(FeaturesBase):
    """キーワードが何個あるかの特徴量."""

    def transform(self, df):
        df = preprocess_keywords(df)
        df["keywords_len"] = df["keywords"].str.split(", ").agg(len)
        df.loc[df["keywords"] == "", "keywords_len"] = 0
        return df[["keywords_len"]]


class KeywordsAverageYear(FeaturesBase):
    """キーワードの登場平均時期の平均と、実際の時期の差分を計算する."""

    def fit(self, df):
        df = preprocess_keywords(df)
        keywords_flattened = flatten_keywords(df)
        self.keywords_avg_year_dict = keywords_flattened.groupby("keyword").mean("year").to_dict()["year"]
        return self

    def transform(self, df):
        df = preprocess_keywords(df)
        df["keywords_avg_year"] = df["keywords"].apply(
            apply_mapping_to_keywords, target_mapping=self.keywords_avg_year_dict
        )
        df["diff_years_keywords_avg_and_actual"] = df["keywords_avg_year"] - df["year"]
        return df[["diff_years_keywords_avg_and_actual"]]


class KeywordsAverageCount(FeaturesBase):
    """キーワードの平均登場回数."""

    def fit(self, df):
        df = preprocess_keywords(df)
        keywords_flattened = flatten_keywords(df)
        self.keywords_avg_count_dict = keywords_flattened.groupby("keyword").count().to_dict()["year"]
        return self

    def transform(self, df):
        df = preprocess_keywords(df)
        df["keywords_avg_count"] = df["keywords"].apply(
            apply_mapping_to_keywords, target_mapping=self.keywords_avg_count_dict
        )
        return df[["keywords_avg_count"]]
