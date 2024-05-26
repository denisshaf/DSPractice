from abc import ABC, abstractmethod
import pandas as pd
from typing import NamedTuple


class SplitterConfig(NamedTuple):
    train_size: int
    val_size: int | None
    test_size: int | None


class Splitter(ABC):

    @abstractmethod
    def split(self, data: pd.DataFrame, 
              config: SplitterConfig,
              sorted: bool = True
              ) -> dict[str, pd.DataFrame]:
        pass