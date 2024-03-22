from ETL import ETL
from abc import ABC
import os


class CsvEtl(ETL, ABC):
    def load(self) -> None:
        path = './data/transformed/'
        if not os.path.exists(path):
            os.mkdir(path)
        for name, dataframe in self.result_data.items():
            dataframe.to_csv(f'{path}{name}.csv', index=False)
