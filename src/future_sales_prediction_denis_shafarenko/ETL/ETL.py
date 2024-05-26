import pandas as pd
from abc import ABC, abstractmethod


class ETL(ABC):

    def __init__(self):
        self.raw_data = {}
        self.result_data = {}

    def extract(self, path: str, name : str | None = None):
        if name is None:
            name = path

        self.raw_data[name] = pd.read_csv(path)

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def load(self):
        pass