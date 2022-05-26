from lilac.features.generator_base import FeaturesBase


class ReviewedMonth(FeaturesBase):
    """reviewされた月をnumber_of_reviewsとreviews_per_monthから逆算する."""

    def transform(self, df):
        df["reviewed_month"] = df["number_of_reviews"] / df["reviews_per_month"]
        return df[["reviewed_month"]]


class MinNightsAvailability(FeaturesBase):
    """minimum_nightsとavailability365の和差積商."""

    def transform(self, df):
        df["min_nights_availability_add"] = df["minimum_nights"] + df["availability_365"]
        df["min_nights_availability_sub"] = df["minimum_nights"] - df["availability_365"]
        df["min_nights_availability_prod"] = df["minimum_nights"] * df["availability_365"]
        df["min_nights_availability_div"] = df["minimum_nights"] / df["availability_365"]

        return df[
            [
                "min_nights_availability_add",
                "min_nights_availability_sub",
                "min_nights_availability_prod",
                "min_nights_availability_div",
            ]
        ]


class PreprocessName(FeaturesBase):
    def transform(self, df):
        df["name_preprocessed"] = (
            df["name"]
            .apply(lambda x: " ".join(x.split("/")))
            .apply(lambda x: " ".join(x.split("|")))
            .apply(lambda x: " ".join(x.split(",")))
        )
        return df[["name_preprocessed"]]


class RulebaseName(FeaturesBase):
    def transform(self, df):
        df["word_count"] = df["name_preprocessed"].apply(lambda x: len(x.split()))
        df["name_len"] = df["name_preprocessed"].apply(lambda x: len(x))
        df["is_wifi"] = df["name_preprocessed"].apply(lambda x: "wi-fi" in x.lower() or "wifi" in x.lower())
        df["is_free"] = df["name_preprocessed"].apply(lambda x: "free" in x.lower())
        df["is_min"] = df["name_preprocessed"].apply(lambda x: "min " in x.lower())
        df["is_skytree"] = df["name_preprocessed"].apply(lambda x: "skytree" in x.lower())
        df["is_sale"] = df["name_preprocessed"].apply(lambda x: "sale" in x.lower())
        df["is_star"] = df["name_preprocessed"].apply(lambda x: "★" in x.lower() or "☆" in x.lower())
        return df[["word_count", "name_len", "is_wifi", "is_free", "is_min", "is_skytree", "is_sale", "is_star"]]
