from abc import ABC, abstractmethod
import pandas as pd


class Splitter(ABC):

    @abstractmethod
    def split(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pass