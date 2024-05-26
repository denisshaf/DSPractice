import json
import os
import pandas as pd

from . import (
    MODEL_CONFIG_PATH, PRED_PATH,
    MODEL_PATH, EXTRACTED_DATA_PATH
    )
from .model.CatBoostPredictor import CatBoostPredictor


def parse_cmd_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(prog='Future sales prediction')
    parser.add_argument('-c', '--config', default=MODEL_CONFIG_PATH)
    parser.add_argument('-d', '--data')
    parser.add_argument('-f', '--features', default=os.path.join(EXTRACTED_DATA_PATH, 'features.parquet'))
    parser.add_argument('-l', '--last-month', default=os.path.join(EXTRACTED_DATA_PATH, 'last_month_lags.parquet'))
    parser.add_argument('-m', '--model-path', default=MODEL_PATH)
    parser.add_argument('-m', '--pred-file', default=os.path.join(PRED_PATH, 'submission'))

    return vars(parser.parse_args())


def get_pred_features(data: pd.DataFrame, features: pd.DataFrame, last_month_lags: pd.DataFrame, config: dict) -> pd.DataFrame:
    last_month = features['date_block_num'].max()
    cat_features = features[features['date_block_num'] == last_month][config['categorical_features']]
    pred_features = cat_features.merge(data,
                            on=['shop_id', 'item_id'],
                            how='right')
    
    pred_features['date_block_num'] = last_month + 1
    pred_features['month'] = (last_month + 1) % 12
    pred_features['year'] = (last_month + 1) // 12

    last_month_lags['index'] = 1
    last_month_lags['index'] = last_month_lags['index'].cumsum() - 1
    last_month_lags = last_month_lags.set_index('index')
    last_month_lags = last_month_lags.merge(pred_features[['shop_id', 'item_id']], left_index=True, right_index=True)
    
    pred_features = pred_features.merge(last_month_lags, on=['shop_id', 'item_id'])
    first_lag_columns = [lag['column'] for lag in config['lags'] if lag['lags'][0] == 1]
    for col in first_lag_columns:
        pred_features = pred_features.rename(columns={col: f'{col}_lag_1'})

    lags_dict = {lag['column']: lag['lags'].remove(1) for lag in config['lags']}

    for feature, lags in lags_dict.items():
        for lag in lags:
            pred_features = pred_features.merge(data[data['date_block_num'] == (last_month - lag + 1)][['shop_id', 'item_id', f'{feature}_lag_1']],
                                        on=['shop_id', 'item_id'],
                                        how='left',
                                        suffixes=('', '_right')) \
                                .rename(columns={f'{feature}_lag_1_right': f'{feature}_lag_{lag}'})
            
    return pred_features


if __name__ == '__main__':
    cmd_args = parse_cmd_args()

    config_path = cmd_args['data_path']
    with open(config_path) as json_file:
        config = json.load(json_file)

    data = pd.read_csv(cmd_args['data'])
    features = pd.read_parquet(os.path.join(cmd_args['features']))
    last_month_lags = pd.read_parquet(os.path.join(cmd_args['last_month']))

    pred_features = get_pred_features(data, features, last_month_lags, config)

    model = CatBoostPredictor()
    model.load_model(os.path.join(cmd_args['model_path'], 'model.cbm'))

    y_pred = model.predict(pred_features)
    submission = pd.DataFrame({'ID': data['ID'], config['target']: y_pred})
    submission.to_csv(cmd_args['pred_file'], index=False)
