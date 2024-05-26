from .splitter import Splitter, SplitterConfig
import pandas as pd


class TTVTimeSeriesSplitter(Splitter):
    def split(self, data: pd.DataFrame, 
              config: SplitterConfig,
              sorted_: bool,
              ) -> dict[str, pd.DataFrame]:
        """ Splits data into train, validation and test subset

            data must be sorted by date_block_num

            train_size, val_size, test_size are sizes of subsets in months.
            If all are set, they must sum to the number of all months.
            If val_size is set and test_size is not, test_size is the rest of the months.
            If test_size is set and val_size is not, val_size is the rest of the months.
            If val_size and test_size are not set, they are splitted equally.
        """    

        if sorted_:
            max_month = data['date_block_num'][-1]
        else:
            max_month = data['date_block_num'].max

        train_size, val_size, test_size = config

        if train_size + int(val_size is None) + int(test_size is None):
            raise ValueError(f"train_size, val_size, test_size don't sum to {max_month}")
        
        if val_size is None and test_size is None:
            val_size = (max_month - train_size) >> 1 + (max_month - train_size) & 1
            test_size = (max_month - train_size) >> 1
        elif val_size is None:
            val_size = max_month - train_size - test_size
        elif test_size is None:
            test_size = max_month - train_size - val_size

        if val_size != 0 and test_size == 0:
            raise ValueError('If validation is set, test subset cannot be empty')


        train_data = data[data['date_block_num'] <= train_size - 1]
        if val_size == 0:
            val_data = None
        else:
            val_data = data[data['date_block_num'].between(train_size, train_size + val_size - 1)]
        if test_size == 0:
            test_data = None
        else:
            test_data = data[data['date_block_num'].between(train_size + val_size, train_size + val_size + test_size)]

        return {'train': train_data, 'validation': val_data, 'test': test_data}

