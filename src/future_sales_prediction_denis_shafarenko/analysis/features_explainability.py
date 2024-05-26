import shap
import pandas as pd

from ..model.CatBoostPredictor import CatBoostPredictor


class Explainer:
    def __init__(self, model: CatBoostPredictor, data: pd.DataFrame, config: dict) -> None:
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer(data.drop(columns=config['target'], errors='ignore'))

    def force_plot(self) -> None:
        shap.plots.force(self.shap_values)

    def beeswarm_plot(self) -> None:
        shap.plots.beeswarm(self.shap_values)
