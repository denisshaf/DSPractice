import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class ETL(ABC):

    def __init__(self):
        self.data = {}
        self.result_data = {}

    def extract(self, path, name=None):
        if name is None:
            name = path

        self.data[name] = pd.read_csv(path)

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def load(self):
        pass