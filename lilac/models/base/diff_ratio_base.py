from lilac.models.model_base import RegressorBase


class DiffRegressorBase(RegressorBase):
    """target_col-base_colに対して最適化する."""

    def __init__(self, model, target_col, base_col):
        super().__init__(target_col)
        self.base_col = base_col
        self.model = model

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)

        # preprocess
        train_df[self.target_col] = train_y - train_x[self.base_col].values
        valid_df[self.target_col] = valid_y - valid_x[self.base_col].values
        # ndarrayにするとバグらない
        # TODO なぜか調査する

        return self.model.fit(train_df, valid_df)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        pred = self.model.predict(test_df)
        # ndarrayにするとバグらない
        # TODO なぜか調査する
        h = test_df[self.base_col].values
        return pred + h
