import numpy as np
from .CsvEtl import CsvEtl


class DqcCsvEtl(CsvEtl):

    def transform(self) -> None:

        # drop duplicates
        self.data['sales_train'].drop_duplicates(inplace=True)

        # drop negative values in `item_price`
        self.data['sales_train'].drop(self.data['sales_train'][self.data['sales_train']['item_price'] <= 0].index,
                                      inplace=True)

        # drop outliers in `item_price` and `item_cnt_day`
        self.data['sales_train'].drop((self.data['sales_train'][self.data['sales_train']['item_price'] > 40_000]).index,
                                      inplace=True)
        self.data['sales_train'].drop(self.data['sales_train'][self.data['sales_train']['item_cnt_day'] >= 1000].index,
                                      inplace=True)

        self.result_data = self.data
