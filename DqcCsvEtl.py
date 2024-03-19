import numpy as np
from CsvEtl import CsvEtl


class DqcCsvEtl(CsvEtl):
    def transform(self) -> None:

        # drop negative values in `item_cnt_day` and `item_price`
        self.data['sales_train'].drop(self.data['sales_train'][self.data['sales_train']['item_cnt_day'] <= 0].index,
                                      inplace=True)
        self.data['sales_train'].drop(self.data['sales_train'][self.data['sales_train']['item_price'] <= 0].index,
                                      inplace=True)

        # converting `item_cnt_day` to int
        self.data['sales_train']['item_cnt_day'] = self.data['sales_train']['item_cnt_day'].astype('int')

        # drop outliers in `item_price` and `item_cnt_day`
        sqrt_transformed = self.data['sales_train']['item_price'].apply(np.sqrt)
        q25, q75 = np.quantile(sqrt_transformed, [.25, .75])
        iqd = q75 - q25
        borders = q25 - 1.5 * iqd, q75 + 1.5 * iqd
        self.data['sales_train'].drop(self.data['sales_train'][sqrt_transformed > borders[0]].index,
                                      inplace=True)
        self.data['sales_train'].drop(self.data['sales_train'][self.data['sales_train']['item_cnt_day'] > 5].index,
                                 inplace=True)

        # normalization
        self.data['sales_train']['item_price'] = (self.data['sales_train']['item_price'] - self.data['sales_train'][
            'item_price'].mean()) / self.data['sales_train']['item_price'].std()
        self.data['sales_train']['item_cnt_day'] = (self.data['sales_train']['item_cnt_day'] - self.data['sales_train'][
            'item_cnt_day'].mean()) / self.data['sales_train']['item_cnt_day'].std()