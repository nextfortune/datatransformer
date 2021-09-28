import json
import vaex as vx
import numpy as np
import pandas as pd
import tensorflow as tf

from ast import literal_eval

from datatransformer.abstractobject import DataTransformer

class TensorflowDataTransformer(DataTransformer):
    def __init__(self, data_spec: dict, data={}, feature_column_config={}, *arg, **kwargs):
        """Creates a DataTransformer object.
        Args:
          data_spec: A dictionary that represents the dataset about to be transformed.
          data: if using small sample of data, feed it into the data dictionary to transform it.
          feature_column_config: a dictionary that defines the categories of feature columns
          
          ```python
          data_spec = {
              'foo': {
                  'type': 'non_sequential',
                  'file_path': ["/path/to/data/foo.csv"], # if not set you can also feed data with §data§ arg
                  'dense_feature': ['foo'],
                  'sparse_feature': ['bar']
              }
              'bar': {
                  'type': 'sequential',
                  'file_path': ["/path/to/data/bar.csv"],
                  'dense_feature': ['foo'],
                  'sparse_feature': ['bar']
              }
              'labels': {
                  'type': 'classification', # §classification§ or §regression§
                  'file_path': [/path/to/data/label.csv]
                  'label': ['label']
              }
          }
          feature_column_config = {
              'foo': {
                  'column': ['category1', 'category2' ...]
              }
              'bar':{
                  'column': ['category3', 'category4' ...]
              }
          }
          TensorflowDataTransformer(data_spec=data_spec)
          ```
        """
        self._data_spec = data_spec
        self._data = data
        self._feature_column_config = feature_column_config

        if not data:
            for dim, spec in data_spec.items():
                if 'file_path' not in spec:
                    raise ValueError("Please specify file_path in data_spec if data is not set.")
        else:
            self._data = data.copy()
            self._data_reshape()
        self._load()
        
    @property
    def dimensions(self):
        return [dim for dim in self._data_spec.keys() if dim != 'labels']

    @property
    def dense_features(self):
        return {dim: spec['dense_feature'] for dim, spec in self._data_spec.items()}

    @property
    def sparse_features(self):
        return {dim: spec['sparse_feature'] for dim, spec in self._data_spec.items()}

    @property
    def feature_columns(self):
        self._feature_columns = {}
        for dim in self.dimensions:
            if self._data_spec[dim]['type'] == 'non_sequential':
                self._feature_columns['non_sequential'] = {
                    dim: {
                        'dense': [
                            tf.feature_column.numeric_column(feat) 
                            for feat in self._data_spec[dim]['dense_feature']
                        ],
                        'sparse': [
                            tf.feature_column.categorical_column_with_vocabulary_list(
                                feat, self._feature_column_config[dim][feat]
                            )
                            for feat in self._data_spec[dim]['sparse_feature']
                        ]
                    }
                }
            elif self._data_spec[dim]['type'] == 'sequential':
                self._feature_columns['sequential'] = {
                    dim: {
                        'dense': [
                            tf.feature_column.sequence_numeric_column(feat)
                            for feat in self._data_spec[dim]['dense_feature']
                        ],
                        'sparse': [
                            tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                                feat, self._feature_column_config[dim][feat]
                            )
                            for feat in self._data_spec[dim]['sparse_feature']
                        ]
                    }
                }
            else:
                raise ValueError("Unsupported type {}".format(self._data_spec[dim]['type']))
        return self._feature_columns

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

    def _data_reshape(self):
        for dim in self.dimensions:
            if self._data_spec[dim]['type'] == 'sequential':
                group = self._data[dim].set_index('trans_id').groupby('trans_id')
                #這裡的group 可能要根據data客製化
                for i, g in group:
                    if len(g) > 1:
                        break
                df = group.agg({col: lambda x: x.tolist() for col in g.columns}, axis=1).reset_index()
                self._data[dim] = df

    def _load(self):
        if self._data:
            self._data_parser()
        else:
            self._file_parser()

    def _data_parser(self):
        if 'labels' in self._data:
            self._labels = tf.data.Dataset.from_tensor_slices(dict(self._data.pop('labels')))
            self._data.pop('labels')
        else:
            self._labels = None

        for dim, val in self._data.items():
            if self._data_spec[dim]['type'] == 'non_sequential':
                self._data_spec[dim]['data'] = tf.data.Dataset.from_tensor_slices(
                    dict(val[self.sparse_features[dim]+self.dense_features[dim]])
                )
            else:
                self._data_spec[dim]['data'] = tf.data.Dataset.from_tensor_slices({
                    feature: tf.ragged.constant(val[[feature]].values)
                    for feature in self.sparse_features[dim] + self.dense_features[dim]
                })

    def _file_parser(self):
        if 'labels' in self._data_spec:
            self._labels = tf.data.Dataset.from_tensor_slices(
                dict(vx.open(self._data_spec['labels']['file_path']).to_pandas_df())
            )
            self._data_spec.pop('labels')
        else:
            self._labels = None
        
        for dim, spec in self._data_spec.items():
            if spec['type'] == 'sequential':
                converter_dict =dict.fromkeys(
                    spec['dense_feature']+spec['sparse_feature'], lambda x: literal_eval(x)
                )
                vx_frame = vx.open(
                    path=spec['file_path'], converters=converter_dict
                )
                self._data_spec[dim]['data'] = tf.data.Dataset.from_tensor_slices({
                    x: (lambda x : tf.ragged.constant(vx_frame[x].to_numpy()))(x)
                    for x in spec['dense_feature']+spec['sparse_feature']
                })
            elif spec['type'] == 'non_sequential':
                self._data_spec[dim]['data'] = tf.data.experimental.make_csv_dataset(
                    file_pattern=spec['file_path'],
                    select_columns=spec['dense_feature']+spec['sparse_feature'],
                    header=True, batch_size=1
                )
            else:
                raise ValueError("the dimension type should be either sequential or non_sequential.")

    def list_files(self):
        return {dim: spec['file_path'] for dim, spec in self._data_spec.items()}

    def to_dataset(self, shuffle=False, batch_size=1):
        features = tf.data.Dataset.zip(
            tuple(spec['data'] for dim, spec in self._data_spec.items())
        )
        
        if self._labels is not None:
            ds = tf.data.Dataset.zip((features, self._labels))
        else:
            ds = features

        if shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        if batch_size:
            ds = ds.batch(batch_size)
        return ds
