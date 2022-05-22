"""mainスクリプト."""
import pandas as pd
from custom.feature_generators import ReviewedMonth, MinNightsAvailability, PreprocessName, RulebaseName
from lilac.core.blocks import BlocksRunner


target_col = "y"
evaluator_flag = "rmsle"
features_dir = "data/features"
seed = 42
trainer_str = "basic"
fold_gen_str = "group"
fold_num = 5


trainer_factory_settings = {
    "model_str": trainer_str,
    "params": {
        "target_col": target_col,
        "seed": seed
    }
}

folds_gen_factory_settings = {
    "model_str": "group",  # kfold, stratified, group, stratified_group
    "params": {
        "fold_num": fold_num,
        "seed": seed,
        "target_col": "y",
        "key_col": "host_id"
    }
}

model_factory_settings = {
    "model_str": "lgbm_rmsle",
    "params": {
        "depth": 5,
        "n_estimators": 100000,
        "seed": seed
    }
}

custom_members = {
    "reviewed_month": ReviewedMonth,
    "min_nights_availability": MinNightsAvailability,
    "rulebase_name": RulebaseName,
    "preprocess_name":  PreprocessName
}

unused_cols = ["id", "host_id", "name", "last_review", "last_review_day"]

settings = [
    # 自然言語処理
    {"model_str": "datetime", "params": {"input_col": "last_review"}},
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "wc_vec", "params": {"input_col": "name_preprocessed",
                                                   "vectorizer_str": "tfidf", "decomposer_str": "svd", "n_components": 32, "random_state": 42}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "wc_vec", "params": {"input_col": "name_preprocessed",
                                                   "vectorizer_str": "tfidf", "decomposer_str": "nmf", "n_components": 10, "random_state": 42}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "wc_vec", "params": {"input_col": "name_preprocessed",
                                                   "vectorizer_str": "bow", "decomposer_str": "svd", "n_components": 32, "random_state": 42}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "wc_vec", "params": {"input_col": "name_preprocessed",
                                                   "vectorizer_str": "bow", "decomposer_str": "nmf", "n_components": 10, "random_state": 42}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "use_prev_only": True,
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "bert", "params": {"model_name": "bert-base-multilingual-uncased",
                                                 "input_col": "name_preprocessed",  "max_len": 128}},
                {"model_str": "dec", "params": {"decomposer_str": "pca",
                                                "n_components": 20, "random_state": 42, "col_mark": "bert"}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "use_prev_only": True,
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "bert", "params": {"model_name": "bert-base-multilingual-uncased",
                                                 "input_col": "name_preprocessed",  "max_len": 128}},
                {"model_str": "dec", "params": {"decomposer_str": "umap",
                                                "n_components": 10, "random_state": 42, "col_mark": "bert"}},
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "preprocess_name"},
                {"model_str": "rulebase_name"}
            ]
        }
    },
    # 駅関連
    {"model_str": "extra_join", "params": {
        "csv_path": "data/nearest_station.csv", "join_on": "id"}},
    {"model_str": "extra_join", "params": {
        "csv_path": "data/dist_under_counts.csv", "join_on": "id"}},
    {"model_str": "extra_join", "params": {
        "csv_path": "data/nearest_terminal.csv", "join_on": "id"}},
    {"model_str": "cluster", "params": {"input_cols": [
        "latitude", "longitude"], "model_str":"gmm", "n_clusters":10, "random_state":42}},
    {"model_str": "cluster", "params": {"input_cols": [
        "latitude", "longitude"], "model_str":"kmeans", "n_clusters":10, "random_state":42}},
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "extra_join", "params": {
                    "csv_path": "data/nearest_station.csv", "join_on": "id"}},
                {"model_str": "cluster", "params": {"input_cols": [
                    "sta_latitude", "sta_longitude"], "model_str":"kmeans", "n_clusters":10, "random_state":42}}
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "feature_generators": [
                {"model_str": "extra_join", "params": {
                    "csv_path": "data/nearest_station.csv", "join_on": "id"}},
                {"model_str": "cluster", "params": {"input_cols": [
                    "sta_latitude", "sta_longitude"], "model_str":"gmm", "n_clusters":10, "random_state":42}}
            ]
        }
    },
    {
        "model_str": "pipeline",
        "params": {
            "use_prev_only": [False, True],
            "feature_generators":[
                {"model_str": "extra_join", "params": {
                    "csv_path": "data/nearest_station.csv", "join_on": "id"}},
                {"model_str": "group", "params": {"group_key": "station_name", "input_cols": [
                    "minimum_nights", "number_of_reviews", "availability_365"]}},
                {"model_str": "dec", "params": {"decomposer_str": "umap",
                                                "n_components": 10, "random_state": 42, "col_mark": "station_group"}},
            ]
        }
    },
    # その他
    {"model_str": "reviewed_month"}
]
train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")

register_from = "custom.feature_generators"
blocks_runner = BlocksRunner(target_col=target_col, features_dir=features_dir, register_from=register_from, features_settings=settings,
                             unused_cols=unused_cols, folds_gen_factory_settings=folds_gen_factory_settings, model_factory_settings=model_factory_settings,
                             trainer_factory_settings=trainer_factory_settings, evaluator_str=evaluator_flag)

output = blocks_runner.run(train, test)

print(output["score"])
