from DqcCsvEtl import DqcCsvEtl
import os

if __name__ == '__main__':
    etl = DqcCsvEtl()
    raw_data_path = './data/raw'
    for file in os.listdir(raw_data_path):
        if file.endswith('.csv'):
            etl.extract(f'{raw_data_path}/{file}', name=file[:-4])
    etl.transform()
    etl.load()


