"""Default values for hyperparameters of models."""

early_stopping_rounds = 100
depth = 5
learning_rate = 0.1
n_estimators = 10000
seed = None
class_weight = "balanced"

# lgbm
verbose_eval = 100
colsample_bytree = 0.8
reg_alpha = 0
reg_lambda = 0
subsample = 0.8
min_child_weight = 1.0


# catb
num_leaves = int(2**depth * 0.7)
random_strength = (1,)
bagging_temperature = (0.1,)
od_type = ("IncToDec",)
od_wait = (10,)
