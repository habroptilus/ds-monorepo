from lilac.jobs.job_base import BasicSeedJob

if __name__ == "__main__":
    target_col = "y"
    evaluator_flag = "rmsle"
    features_dir = "data/features"
    seed = 42
    trainer_str = "basic"
    fold_gen_str = "group"
    fold_num = 5
    depth = 5
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"
    group_key_col = "host_id"
    model_str = "lgbm_rmsle"
    folds_gen_str = "group"
    evaluator_str = "rmsle"
    unused_cols = ["id", "host_id", "name",
                   "last_review", "last_review_day"]
    n_estimators = 2000
    features_settings = [
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

    register_from = "custom.feature_generators"
    output = BasicSeedJob(train_path, test_path, features_dir, register_from, features_settings, target_col, unused_cols, seed,
                          fold_num, group_key_col,  depth, n_estimators, trainer_str, model_str, folds_gen_str, evaluator_str).run()

    print(output["score"])
