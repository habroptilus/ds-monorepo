"""Default values for hyperparameters of models."""

early_stopping_rounds = 100
depth = 5
learning_rate = 0.1
n_estimators = 40000
seed = None
class_weight = None  # "balanced"


# lgbm
verbose_eval = 100
colsample_bytree = 1.0
reg_alpha = 0
reg_lambda = 1
subsample = 0.9
min_child_weight = 1.0
min_child_samples = 20


# catb
random_strength = 1
bagging_temperature = 0.1
od_type = "IncToDec"
od_wait = 10
