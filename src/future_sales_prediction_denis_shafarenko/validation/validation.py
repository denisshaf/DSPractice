import pandas as pd
import tensorflow_data_validation as tfdv
import tensorflow_metadata as tfmd
from tensorflow_metadata.proto.v0.anomalies_pb2 import Anomalies
from typing import Literal, TypedDict

from .schema_writer import load_schema
from ..utils.type_conversion import squeeze_int, squeeze_float
from .exceptions import InvalidSchemaException, StatisticsAnomaly, DriftAnomaly
from ..splitter.splitter import Splitter, SplitterConfig
from .feature_type import FeatureType 


class ValidationResult(TypedDict):
    stats: dict
    drift: dict
    skew: dict | None


class Validator:
    def __init__(self,
                 splitter: Splitter,
                 schema_path: str | None = None):
        if schema_path is not None:
            self.schema = load_schema(schema_path)

        self.splitter = splitter
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_stats = None
        self.val_stats = None
        self.test_stats = None

        self.optional_features = []


    def run_validation(self,
                    data: pd.DataFrame,
                    splitter_config: SplitterConfig,
                    drift_comparator_config: dict[str, tuple[Literal['infinity_norm', 'jensen_shannon_divergence'], float]],
                    display_anomalies: bool = False,
                    raise_exception: bool = True,
                    opt_features: list[str] | None = None) -> None:
        """ Runs validation piplene. Checks statistics and drift anomalies. 
            If there are anomalies, raises exception with anomaly object. If not, returns None
        """

        if splitter_config.test_size == 0:
            return

        data = self.preprocess_types(data)

        split = self.splitter.split(data, splitter_config, sorted=True)
        self.train_data = split['train']
        self.val_data = split.get('validation')
        self.test_data = split['test']

        if opt_features is not None:
            self.add_optional_features(opt_features)

        self.generate_statistics()
        stats_anomalies = self.validate_statistics()

        for key, anomaly in stats_anomalies.items():
            if anomaly.anomaly_info:
                if display_anomalies:
                    tfdv.display_anomalies(anomaly)
                if raise_exception:
                    raise StatisticsAnomaly(sample_name=key, anomaly=anomaly)

        drift_anomalies = dict()
        if self.val_data is None:
            drift_anomalies['train_test'] = self.check_drift(drift_comparator_config,
                                                             between=('train', 'test'))
        else:
            drift_anomalies['train_val'] = self.check_drift(drift_comparator_config,
                                                             between=('train', 'validation'))
            drift_anomalies['val_test'] = self.check_drift(drift_comparator_config,
                                                            between=('validation', 'test'))
        
        for key, anomaly in drift_anomalies.items():
            if anomaly.anomaly_info:
                if display_anomalies:
                    tfdv.display_anomalies(anomaly)
                if raise_exception:
                    raise DriftAnomaly(sample_name=key, anomaly=anomaly)


    def preprocess_types(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            self.schema = tfdv.infer_schema(data)

        try:
            features = filter(lambda f: f.name not in self.optional_features, self.schema.feature)
            for feature in features:
                match feature.type:
                    case FeatureType.INT:
                        data[feature.name] = data[feature.name].astype('int')
                        data[feature.name] = squeeze_int(data[feature.name])
                    case FeatureType.FLOAT:
                        data[feature.name] = data[feature.name].astype('float')
                        data[feature.name] = squeeze_float(data[feature.name])
                    case FeatureType.BYTES:
                        if feature.domain == 'datetime':
                            data[feature.name] = pd.to_datetime(data[feature.name])
                        else:
                            data[feature.name] = data[feature.name].astype('object')
                    case _:
                        raise InvalidSchemaException(f'Invalid feature type: {feature.type.name}')

        except KeyError as e:
            raise InvalidSchemaException(f'No such feature in data: {e.args[0]}') from None

        return data


    def generate_statistics(self) -> None:
        stats_options = tfdv.StatsOptions(schema=self.schema)

        self.train_stats = tfdv.generate_statistics_from_dataframe(self.train_data, stats_options)
        if self.val_data is not None:
            self.val_stats = tfdv.generate_statistics_from_dataframe(self.val_data, stats_options)
        self.test_stats = tfdv.generate_statistics_from_dataframe(self.test_data, stats_options)


    def add_optional_features(self, opt_features: list[str]) -> None:
        self.optional_features.extend(opt_features)

        if 'TRAINING' not in self.schema.default_environment:
            self.schema.default_environment.append('TRAINING')
        if 'SERVING' not in self.schema.default_environment:
            self.schema.default_environment.append('SERVING')

        # not_in_serving_features = ['item_name',
        #                            'item_category_id',
        #                            'item_category_name',
        #                            'date',
        #                            'date_block_num',
        #                            'item_price',
        #                            'item_cnt_day',
        #                            'shop_name']

        for feature in opt_features:
            tfdv.get_feature(self.schema, feature).not_in_environment.append('SERVING')


    def validate_statistics(self) -> dict[Literal['train', 'validation', 'test'], Anomalies]:
        train_anomalies = tfdv.validate_statistics(statistics=self.train_stats, schema=self.schema)
        test_anomalies = tfdv.validate_statistics(statistics=self.test_stats, schema=self.schema)
        anomalies = {'train': train_anomalies, 'test': test_anomalies}

        if self.val_stats:
            val_anomalies = tfdv.validate_statistics(statistics=self.val_stats, schema=self.schema)
            anomalies['validation'] = val_anomalies

        return anomalies


    def check_drift(self,
                    drift_comparator_config: dict[str, tuple[Literal['infinity_norm', 'jensen_shannon_divergence'], float]],
                    between: tuple[Literal['train', 'validation', 'test'],
                                   Literal['train', 'validation', 'test']]
                    ) -> Anomalies:

        features = filter(lambda f: f.name not in self.optional_features, self.schema.feature)
        features_dict = {f.name: f.type for f in features}
        for feature, (metric, threshold) in drift_comparator_config.items():
            if features_dict[feature] in [FeatureType.INT, FeatureType.FLOAT] and metric == 'infinity_norm':
                raise TypeError(f'Infinity norm is not supported for numerical types: '
                                f'{feature} has type {features_dict[feature].name}')
            tfdv.get_feature(self.schema, feature).drift_comparator.getattr(metric).threshold = threshold

        if self.val_stats is None and 'validation' in between:
            raise ValueError('Validation set is not defined')

        prev_stats = self.train_stats if 'train' in between else self.val_stats
        stats = self.test_stats if 'test' in between else self.val_stats

        anomalies = tfdv.validate_statistics(previous_statistics=prev_stats,
                                             statistics=stats,
                                             schema=self.schema)
        return anomalies


    def check_skew(self):
        raise NotImplementedError()
