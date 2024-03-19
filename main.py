from DqcCsvEtl import DqcCsvEtl
import os

if __name__ == '__main__':
    etl = DqcCsvEtl()
    for file in os.listdir('./data'):
        etl.extract(f'./data/{file}', name=file[:-4])
    etl.transform()
    etl.load()


