from enum import Enum, auto


class FeatureType(Enum):
    TYPE_UNKNOWN = auto()
    BYTES = auto()
    INT = auto()
    FLOAT = auto()
    STRUCT = auto()