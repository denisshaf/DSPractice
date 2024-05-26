import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..model.CatBoostPredictor import CatBoostPredictor


class ErrorAnalizer:
    def __init__(self, model: CatBoostPredictor, data: pd.DataFrame, config: dict) -> None:
        self.config = config
        self.data = data

        X = data.drop(columns=config['target'])
        y = data[config['target']]
        y_pred = model.predict(X)
        self.data['pred_error'] = np.abs(y - y_pred)

    def plot_categorical_error(self, feature: str) -> None:
        print(self.data.groupby(feature)['pred_error'].mean().reset_index().to_string())
        sns.catplot(data=self.data, x=feature, y="pred_error", kind="box")
        plt.show()

    def plot_numeric_error(self, feature: str, bins: int) -> None:
        self.data[f'{feature}_bins'] = list(pd.cut(self.data['feature'], bins, retbins=False, labels=range(bins)))

        print(self.data.groupby(f'{feature}_bins')['pred_error'].mean().reset_index().to_string())
        sns.catplot(data=self.data, x=f'{feature}_bins', y="pred_error", kind="box")
        plt.show()