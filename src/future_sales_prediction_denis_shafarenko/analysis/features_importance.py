import matplotlib.pyplot as plt
import numpy as np

from ..model.CatBoostPredictor import CatBoostPredictor


def plot_feature_importance(model: CatBoostPredictor, config: dict, save_file: str | None = None) -> None:
    feature_importance = model.get_feature_importance()
    sorted_idx = np.argsort(feature_importance)

    fig = plt.figure(figsize=(12, 6))
    
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(data.drop(columns=config['target']).columns)[sorted_idx])
    plt.title('Feature Importance')

    if save_file is not None:
        plt.savefig(save_file)

    plt.show()