import numpy as np
import pandas as pd
import tensorflow as tf

from datatransformer.abstractobject import DataTransformer

class TensorflowDataTransformer(DataTransformer):
    def __init__(self, data: dict, data_spec: dict, *arg, **kwargs):
        if 'labels' in data:
            self._labels = data.pop('labels')

        self._data = data.copy()
        self._data_spec = data_spec
        self._data_reshape()

    @property
    def dimensions(self):
        if self._data.keys() == self._data_spec.keys():
            return list(self._data.keys())
        else:
            raise ValueError('dimensions between data and data_spec must be equal.')

    @property
    def dense_features(self):
        return {dim: spec['dense_feature'] for dim, spec in self._data_spec.items()}

    @property
    def sparse_features(self):
        return {dim: spec['sparse_feature'] for dim, spec in self._data_spec.items()}

    @property
    def feature_columns(self):
        feature_columns = {}
        for dim in self.dimensions:
            if self._data_spec[dim]['type'] == 'non_sequential':
                feature_columns[dim] = []
                for feat in self._data_spec[dim]['sparse_feature']:
                    fc = tf.feature_column.categorical_column_with_vocabulary_list(
                        feat, list(self._data[dim][feat].unique())
                    )
                    feature_columns[dim].append(fc)

                for feat in self._data_spec[dim]['dense_feature']:
                    fc = tf.feature_column.numeric_column(feat)
                    feature_columns[dim].append(fc)
            else:
                feature_columns[dim] = []
                for feat in self._data_spec[dim]['sparse_feature']:
                    sparse_union = set()
                    for i in range(len(self._data[dim][feat])):
                        sparse_union = sparse_union.union(self._data[dim][feat][i])
                    fc = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                        feat, sparse_union
                    )
                    feature_columns[dim].append(fc)

                for feat in self._data_spec[dim]['dense_feature']:
                    fc = tf.feature_column.sequence_numeric_column(feat)
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

    def to_dataset(self, shuffle=False, batch_size=1):
        dim_list = tuple()
        for dim, val in self._data.items():
            df_dim = self._data[dim]
            sparse_dims = []
            dense_dims = []
            if self._data_spec[dim]['type'] == 'non_sequential':    
                # extract 'sparse' feature in non_sequential dims
                sparse_feature = self._data_spec[dim]['sparse_feature']
                dim_sparse = df_dim[sparse_feature]
                sparse_dims.append(dict(dim_sparse))

                # extract 'dense' feature in non_sequential dims
                dense_feature = self._data_spec[dim]['dense_feature']
                dim_dense = df_dim[dense_feature]
                dense_dims.append(dict(dim_dense))

                dim_list = dim_list + (tuple(sparse_dims+dense_dims),)
            else:
                sparse_dic = dict()
                dense_dic = dict()
                #create sequece data raggertensor
                col_list = dict()
                for col in df_dim.columns:
                    element = tf.ragged.constant(df_dim[[col]].values)
                    col_list[col] = element

                sparse_feature = self._data_spec[dim]['sparse_feature']
                dense_feature = self._data_spec[dim]['dense_feature']
                # extract 'sparse' feature in sequential dims
                sparse_dic = {k: v for k, v in col_list.items() if k in sparse_feature}
                # extract 'dense' feature in sequential dims
                dense_dic = {k: v for k, v in col_list.items() if k in dense_feature}
                sparse_dims.append(dict(sparse_dic))
                dense_dims.append(dict(dense_dic))

                dim_list = dim_list + (tuple(sparse_dims+dense_dims),)
        if self._labels is not None:
            labels = dict(self._labels)
            ds = tf.data.Dataset.from_tensor_slices((dim_list, labels))
        else:
            ds = tf.data.Dataset.from_tensor_slices(dim_list)

        if shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        if batch_size:
            ds = ds.batch(batch_size)
        return ds
