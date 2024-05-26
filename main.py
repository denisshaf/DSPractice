import os

from src.future_sales_prediction.ETL.DqcCsvEtl import DqcCsvEtl
from src.future_sales_prediction.validation.validation import Validator
from src.future_sales_prediction.splitter.time_series_splitter import TTVTimeSeriesSplitter
from src.future_sales_prediction.splitter.splitter import SplitterConfig
import src.future_sales_prediction.feature_extraction.feature_extraction as fe


if __name__ == '__main__':
    RAW_DATA_PATH = 'src/data/raw'
    TRANSFORMED_DATA_PATH = 'src/data/transformed/'
    EXTRACTED_DATA_PATH = 'src/data/transformed/'
    SCHEMA_PATH = 'src/data/schema/pre_features_schema.pbtxt'
    
    etl = DqcCsvEtl()
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith('.csv'):
            etl.extract(os.path.join(RAW_DATA_PATH, file), name=file[:-4])
    etl.transform()
    etl.load(TRANSFORMED_DATA_PATH)

    data = etl.result_data

    splitter = TTVTimeSeriesSplitter()
    splitter_congfig = SplitterConfig(32, 1, 1)
    drift_comparator_config = {}
    validator = Validator(data, splitter, SCHEMA_PATH)
    validator.run_validation(data, splitter_congfig, drift_comparator_config)

    fe.run_pipeline(TRANSFORMED_DATA_PATH, EXTRACTED_DATA_PATH)

    import matplotlib 
    print(matplotlib.get_backend())




    



