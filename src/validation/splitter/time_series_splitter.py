from .splitter import Splitter
import pandas as pd
from typing import Dict, Literal


class TTVTimeSeriesSplitter(Splitter):
    def split(self, data: pd.DataFrame) -> Dict[str: pd.DataFrame]:
        train_data = data[data['date_block_num'] <= 31]
        val_data = data[data['date_block_num'] == 32]
        test_data = data[data['date_block_num'] == 33]

        return {'train': train_data, 'validation': val_data, 'test': test_data}

