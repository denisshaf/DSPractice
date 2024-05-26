import pandas as pd
import numpy as np


def squeeze_int(s: pd.Series) -> pd.Series:
    max_val = max(abs(s.max), abs(s.min))
    types = ['int8', 'int16', 'int32']

    for type_ in types:
        if max_val <= np.iinfo(type_).max:
            return s.astype(type_)


def squeeze_float(s: pd.Series) -> pd.Series:
    max_val = max(abs(s.max), abs(s.min))

    # float16 isn't fully supported by processors and therefore numpy
    types = ['float32']

    for type_ in types:
        if max_val <= np.finfo(type_).max:
            return s.astype(type_)