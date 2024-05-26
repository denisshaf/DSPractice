import os
import json

from . import (
    RAW_DATA_PATH, TRANSFORMED_DATA_PATH, 
    EXTRACTED_DATA_PATH, SCHEMA_PATH, 
    MODEL_CONFIG_PATH, DATA_PATH,
    MODEL_PATH
    )
from .ETL.DqcCsvEtl import DqcCsvEtl
from .validation.validation import Validator
from .splitter.time_series_splitter import TTVTimeSeriesSplitter
from .splitter.splitter import SplitterConfig
from .feature_extraction import feature_extraction as fe
from .model.CatBoostPredictor import CatBoostPredictor


def parse_cmd_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(prog='Future sales prediction')
    parser.add_argument('-c', '--config', default=MODEL_CONFIG_PATH)
    parser.add_argument('--data-path', default=DATA_PATH)
    parser.add_argument('--model-path', default=MODEL_PATH)

    return vars(parser.parse_args())


if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    if cmd_args['data_path'] != DATA_PATH:
        DATA_PATH = cmd_args['data_path']
        RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
        TRANSFORMED_DATA_PATH = os.path.join(DATA_PATH, 'transformed')
        EXTRACTED_DATA_PATH = os.path.join(DATA_PATH, 'extracted')
        SCHEMA_PATH = os.path.join(DATA_PATH, 'schema')

    config_path = cmd_args['data_path']
    with open(config_path) as json_file:
        config = json.load(json_file)


    etl = DqcCsvEtl()
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            etl.extract(os.path.join(RAW_DATA_PATH, file), name=file[:-4])
    etl.transform()
    etl.load(TRANSFORMED_DATA_PATH)
    data = etl.result_data

    splitter = TTVTimeSeriesSplitter()
    splitter_congfig = SplitterConfig(*config['splitter'])
    drift_comparator_config = config['drift_comparator']
    validator = Validator(splitter, SCHEMA_PATH)
    validator.run_validation(data, splitter_congfig, drift_comparator_config)

    features = fe.run_pipeline(TRANSFORMED_DATA_PATH, EXTRACTED_DATA_PATH, config)


    model = CatBoostPredictor(**config['model'])
    model.fit(features, config['target'], config['categorical_features'], config['numeric_features'])
    model.save_model(os.path.join(cmd_args['model_path'], 'model.cbm'), format='cbm')