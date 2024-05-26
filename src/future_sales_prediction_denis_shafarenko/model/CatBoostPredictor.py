import catboost as ctb
import pandas as pd
from sklearn.metrics import root_mean_squared_error

class CatBoostPredictor(ctb.CatBoostRegressor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cat_features = None
        self.num_features = None

    def fit(self, train_data: pd.DataFrame, target: str, cat_features: list[str], num_features: list[str] | None = None) -> None:
        self.data = train_data
        self.cat_features = cat_features
        self.target = target

        train_data[cat_features] = train_data[cat_features].astype('category')
        if num_features is None:
            num_features = list(set(train_data.columns) - set(cat_features) - set([target]))
        self.num_features = num_features

        super().fit(X=train_data[cat_features + num_features],
                    y=train_data[num_features])
        
    def test(self, test_data: pd.DataFrame) -> float:
        y_val_pred = super().predict(test_data[self.cat_features])
        y_val = test_data[self.target]
        return root_mean_squared_error(y_val, y_val_pred)
