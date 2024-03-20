from ETL import ETL
from abc import ABC
import os


class CsvEtl(ETL, ABC):
    def load(self) -> None:
        dir = './data/transformed/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        for name, dataframe in self.data.items():
            dataframe.to_csv(f'{dir}{name}.csv')
