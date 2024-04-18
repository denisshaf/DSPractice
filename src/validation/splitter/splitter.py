from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class Splitter(ABC):

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Dict[str: pd.DataFrame]:
        pass