import numpy as np
import pandas as pd
import tensorflow as tf

from datatransformer.abstractobject import DataTransformer

class TensorflowDataTransformer(DataTransformer):
    def __init__(self, data: dict, data_spec: dict, *arg, **kwargs):
        if 'labels' in data:
            self._labels = data.pop('labels')
        self._data = data
        self._data_spec = data_spec

    @property
    def dimensions(self):
        if self._data.keys() == self._data_spec.keys():
            return list(self._data.keys())
        else:
            raise ValueError('dimensions between data and data_spec must be equal.')

    @property
    def dense_features(self):
        return sum([spec.get('dense_feature', []) for dim, spec in self._data_spec.items()], [])

    @property
    def sparse_features(self):
        return sum([spec.get('sparse_feature', []) for dim, spec in self._data_spec.items()], [])

    @property
    def feature_columns(self):
        feature_columns = {}
        for dim in self.dimensions:
            feature_columns[dim] = []
            for feat in self._data_spec[dim]['sparse_feature']:
                fc = tf.feature_column.categorical_column_with_vocabulary_list(
                    feat, list(self._data[dim][feat].unique())
                )
                feature_columns[dim].append(fc)

            for feat in self._data_spec[dim]['dense_feature']:
                fc = tf.feature_column.numeric_column(feat)
                feature_columns[dim].append(fc)
        return feature_columns

    @property
    def labels(self):
        return self._labels if hasattr(self, '_labels') else None

    @property
    def buffer_size(self):
        den = iter(self.dense_features)
        len_den = len(next(den))
        if not all(len(l) == len_den for l in den):
            raise ValueError('not all dense feature in same length.')
        return len_den

    def to_dataset(self, shuffle=True, batch_size=32):
        sparse_dims = []
        dense_dims = []
        for dim, val in self._data.items():
            df_dim = self._data[dim]
            # extract 'sparse' feature in every dims
            sparse_feature = self._data_spec[dim]['sparse_feature']
            dim_sparse = df_dim[sparse_feature]
            sparse_dims.append(dict(dim_sparse))

            # extract 'dense' feature in every dims
            dense_feature = self._data_spec[dim]['dense_feature']
            dim_dense = df_dim[dense_feature]
            dense_dims.append(dict(dim_dense))

        dims = (tuple(sparse_dims), tuple(dense_dims),)
        if 'labels' in self._data:
            labels = dict(self._data.get('labels', None))
            ds = tf.data.Dataset.from_tensor_slices(dims, labels)
        else:
            ds = tf.data.Dataset.from_tensor_slices(dims)
        
        if shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        if batch_size:
            ds = ds.batch(batch_size)
        return ds
