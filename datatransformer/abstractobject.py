import abc

class DataTransformer(abc.ABC):

    @property
    @abc.abstractclassmethod
    def dimensions(cls):
        pass
    
    @property
    @abc.abstractclassmethod
    def dense_features(cls):
        pass
    
    @property
    @abc.abstractclassmethod
    def sparse_features(cls):
        pass
    
    @property
    @abc.abstractclassmethod
    def feature_columns(cls):
        pass
    
    @property
    @abc.abstractclassmethod
    def labels(cls):
        pass
    
    @abc.abstractclassmethod
    def to_dataset(cls):
        pass
