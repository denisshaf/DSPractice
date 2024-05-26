import tensorflow_metadata as tfmd


class InvalidSchemaException(Exception):
    pass


class AnomalyException(Exception):
    def __init__(self, *args, sample_name: str, anomaly: tfmd.proto.anomalies_pb2.Anomalies):
        super().__init__(*args)

        self.sample_name = sample_name
        self.anomaly = anomaly


class DriftAnomaly(AnomalyException):
    pass


class StatisticsAnomaly(AnomalyException):
    pass