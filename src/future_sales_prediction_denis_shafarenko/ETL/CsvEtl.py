from .ETL import ETL
from abc import ABC
import os


class CsvEtl(ETL, ABC):
    def load(self, path) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            for file in os.listdir(path):
                os.unlink(os.path.join(path, file))
                
        for name, dataframe in self.result_data.items():
            dataframe.to_csv(os.path.join(path, f'{name}.csv'), index=False)
