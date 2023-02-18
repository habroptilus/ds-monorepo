import pandas as pd


def preprocess_keywords(df):
    """keywordカラムを前処理する.

    - Nullを空文字で置換
    - 小文字
    - ハイフンを空白に置換
    """
    df = df.copy()
    df["keywords"] = df["keywords"].fillna("").astype(str)
    df["keywords"] = df["keywords"].str.lower().str.replace("-", " ")
    return df


def flatten_keywords(df):
    """keywordsをkeywordごとに別の行にバラす.

    keywordとyearの列を持ったdataframeをreturnする.
    (他の列も持ってもいいかも?)
    """
    result = []
    for idx in df.index:
        keywords = df.loc[idx, "keywords"].split(", ")
        year = df.loc[idx, "year"]
        for keyword in keywords:
            result.append({"keyword": keyword, "year": year})
    return pd.DataFrame(result)
