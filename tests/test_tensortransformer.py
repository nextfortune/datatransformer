"""Unit test Tensorflow transformer"""
import unittest
import tensorflow as tf
import generate_fake_data

class Testtensorflowtransformer(unittest.TestCase):
    """Test tensorflow transformer"""
    def setUp(self):
        super().setUp()

        self.datatrans = generate_fake_data.generate_fake_data()
        self.dataset = self.datatrans.to_dataset()
        self.feature_columns = self.datatrans.feature_columns

    def test_dataset_elementspec(self):
        """Test data input element spec"""
        excepted_ele_spec = (
            ({
                'f_feature_0': tf.TensorSpec(
                    shape=(None,), dtype=tf.int64, name=None
                ),
                'f_feature_1': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_2': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_4': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_5': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_3': tf.TensorSpec(
                    shape=(None,), dtype=tf.float64, name=None
                )
            },
            {
                'b_feature_1': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_2': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_3': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_4': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_5': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.float32, 2, tf.int64
                )
            }),
            {
                'output_1': tf.TensorSpec(shape=(None,), dtype=tf.bool, name=None)
            }
        )

        self.assertEqual(excepted_ele_spec, self.dataset.element_spec)

    def test_features_types(self):
        """Test data input features types"""
        sparse = self.datatrans.sparse_features
        excepted_sparse = {
            'foo': [
                'f_feature_0', 'f_feature_1', 'f_feature_2',
                'f_feature_4', 'f_feature_5',
            ],
            'bar': [
                'b_feature_1', 'b_feature_2',
                'b_feature_3', 'b_feature_4'
            ]
        }

        dense = self.datatrans.dense_features
        excepted_dense = {
            'foo': ['f_feature_3'],
            'bar': ['b_feature_5']
        }

        self.assertEqual(excepted_sparse, sparse)
        self.assertEqual(excepted_dense, dense)

    def test_dimensions(self):
        """Test data input dimensions"""
        dim = self.datatrans.dimensions
        excepted_dim = ['foo', 'bar']

        self.assertEqual(excepted_dim, dim)

    def test_dataset_featurecolumns(self):
        """Test feature column of datatransformer"""
        excepted_feature_columns={
            'non_sequential': {
                'foo': {
                    'dense': [
                        tf.feature_column.numeric_column(
                            key='f_feature_3', shape=(1,),
                            default_value=None, dtype=tf.float32, normalizer_fn=None
                        )],
                    'sparse': [
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_0', vocabulary_list=(0, 1, 2, 3, 4),
                            dtype=tf.int64, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_1',
                            vocabulary_list=(
                                'JP', 'HK', 'IN', 'ID', 'NL',
                                'KH', 'CN', 'TH', 'MY', 'VN'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_2',
                            vocabulary_list=(
                                '2021-08-08 08:15:54.550', '2021-07-30 03:45:32.517',
                                '2021-08-18 11:49:47.460', '2021-07-07 12:05:45.600',
                                '2021-08-23 10:53:20.270', '2021-08-02 10:53:11.160',
                                '2021-07-16 12:54:44.997'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_4', vocabulary_list=('False', 'True'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_5', vocabulary_list=('person', 'station', 'song'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        )
                    ]
                }
            },
            'sequential': {
                'bar': {
                    'dense': [
                        tf.feature_column.sequence_numeric_column(
                            key='b_feature_5', shape=(1,), default_value=0.0,
                            dtype=tf.float32, normalizer_fn=None
                        ),
                    ],
                    'sparse': [
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_1',
                            vocabulary_list=(
                                '2021-08-08 08:15:54.550', '2021-07-30 03:45:32.517',
                                '2021-08-18 11:49:47.460', '2021-07-07 12:05:45.600',
                                '2021-08-23 10:53:20.270', '2021-08-02 10:53:11.160',
                                '2021-07-16 12:54:44.997'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_2', vocabulary_list=('False', 'True'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_3', vocabulary_list=('person', 'station', 'song'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_4',
                            vocabulary_list=('a', 'R', 'c', 'j', 'L', 'p', 'K', 'h', 'N', 'v'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        )
                    ]
             }
            }
        }

        self.assertEqual(excepted_feature_columns, self.feature_columns)
