import os

from ETL.DqcCsvEtl import DqcCsvEtl
from validation.validation import Validator
from validation.splitter.time_series_splitter import TTVTimeSeriesSplitter
import feature_extraction.feature_extraction as fe


if __name__ == '__main__':
    RAW_DATA_PATH = './data/raw'
    TRANSFORMED_DATA_PATH = './data/transformed/'
    EXTRACTED_DATA_PATH = './data/transformed/'
    SCHEMA_PATH = 'data/schema/pre_features_schema.pbtxt'
    
    etl = DqcCsvEtl()
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            etl.extract(os.path.join(RAW_DATA_PATH, file), name=file[:-4])
    etl.transform()
    etl.load(TRANSFORMED_DATA_PATH)

    data = etl.result_data

    splitter = TTVTimeSeriesSplitter()
    validator = Validator(data, splitter, SCHEMA_PATH)
    validator.run_validation()

    fe.run_pipeline(TRANSFORMED_DATA_PATH, EXTRACTED_DATA_PATH)




    



