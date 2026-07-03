"""XGBoost on engineered technical features — the practical, hard-to-beat
tabular approach."""

import numpy as np

from .base import Forecaster


class XGBForecaster(Forecaster):
    name = 'XGBoost'

    def __init__(self, **params):
        self.params = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='reg:squarederror',
            n_jobs=4,
        )
        self.params.update(params)
        self.model = None
        self.features = None  # DataFrame from data.add_features, set externally

    def set_features(self, feature_df):
        """Feature matrix aligned with the returns series (index = dates)."""
        self.features = feature_df

    def fit_features(self, train_df):
        import xgboost as xgb

        X = train_df.drop(columns=['target_ret'])
        y = train_df['target_ret']
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, verbose=False)

    def predict_features(self, test_df):
        X = test_df.drop(columns=['target_ret'])
        return self.model.predict(X)
