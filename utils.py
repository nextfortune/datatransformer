import os
import json

import numpy as np
import pandas as pd
import fasttext as ft
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import config

class DataTransformer:
    """This is a class that tranforms data into trainable and predictable form"""
    def __init__(self, data: dict(), feature_colname: list=[]):
        self.data = data
        self.data_trans = self.data["trans"]
        self.data_games = self.data["games"]
        self.feature_colname = feature_colname
        self.idx_true = None
        self.idx_false = None
        self.ftmodel = ft.load_model("game_embedding_model_ep5.bin")
        self.__init_configuration()
        self.user_idx = self.data_trans['trans_id']
        self.__init_predict()

    def __init_configuration(self):
        if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "config.json")):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config.json")) as file:
                self.config_json = json.load(file)
        else:
            self.config_json={'scaler': dict(), 'labelencode': dict()}
        self.scale = self.config_json['scaler']
        self.label_encode = self.config_json['labelencode']
        if self.label_encode is not None:
            for col, cat in self.label_encode.items():
                if col in config.CATGORICAL_FEATURE_LIST:
                    len_before = len(cat)
                    cat = [x for x in cat if x is not None]
                    len_after = len(cat)
                    if len_before != len_after:
                        self.label_encode[col] = cat + [np.nan]

    def __init_predict(self):
        self.data_trans = self.data_trans[self.data_trans.columns[~self.data_trans.columns.isin(config.POP_FEATURE_LIST)]]
        if len(self.feature_colname)==0:
            self.data_trans = self.data_trans
        else:
            self.data_trans = self.data_trans[self.data_trans.columns[self.data_trans.columns.isin(self.feature_colname)]]

    def update_configuration(self):
        for col in self.data_trans.columns:
            scaler_dict = {
                'min': None,
                'max': None,
            }
            if col in config.CATGORICAL_FEATURE_LIST:
                enc_ = LabelEncoder()
                enc_.fit(self.data_trans[col])
                sample_cat = enc_.classes_.tolist()
                len_before = len(sample_cat)
                sample_cat = [x for x in sample_cat if x==x]
                len_after = len(sample_cat)
                if self.label_encode is not None:
                    cat = self.label_encode.get(col)
                else:
                    cat = None
                if cat is None:
                    cat=[]
                else:
                    cat = [x for x in cat if x==x]
                cat = list(set().union(cat,sample_cat))
                cat = sorted(cat)
                self.label_encode[col] = cat
                if len_before != len_after:
                    self.label_encode[col] = cat + [np.nan]
            elif col in config.NUMERICAL_FEATURE_LIST:
                sample_max = max(self.data_trans[col])
                sample_min = min(self.data_trans[col])
                min_max = self.scale.get(col)
                if min_max is None:
                    min_max = scaler_dict
                    min_max['max'] = sample_max
                    min_max['min'] = sample_min
                    self.scale[col] = min_max
                pop_max = min_max['max']
                pop_min = min_max['min']
                if pop_max < sample_max:
                    self.scale[col]['max'] = sample_max
                if pop_min > sample_min:
                    self.scale[col]['min'] = sample_min
        #nan to none on unused feature      
        for col, cat in self.label_encode.items():
            if col in config.CATGORICAL_FEATURE_LIST:
                len_before = len(cat)
                cat = [x for x in cat if x==x]
                len_after = len(cat)
                if len_before != len_after:
                    self.label_encode[col] = cat + [None]
        pop_configdict = dict()
        pop_configdict['labelencode'] = self.label_encode
        pop_configdict['scaler'] = self.scale

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config.json"),'w') as file:
            json.dump(pop_configdict,file,indent=4)

        self.__init_configuration()

    def get_encode_idx(self):
        unknown_idx = []
        for col in self.data_trans.columns:
            if col in self.label_encode:
                cat = self.label_encode[col]
                idx = self.data_trans[~self.data_trans[col].isin(cat)].index
                unknown_idx += idx.tolist()

            elif col in self.scale:
                min_max = self.scale[col]
                scale_max = min_max['max']
                scale_min = min_max['min']
                #check in range
                idx_min = self.data_trans[self.data_trans[col] < scale_min].index
                idx_max = self.data_trans[self.data_trans[col] > scale_max].index
                unknown_idx += idx_min.tolist()
                unknown_idx += idx_max.tolist()
        unknown_idx = list(set(unknown_idx))
        data_idx = self.data_trans.index.tolist()
        predict_idx = list(set(data_idx)-set(unknown_idx))
        return predict_idx, unknown_idx

    def transform(self):
        #trans transform
        if len(self.feature_colname)==0:
            for  col in self.data_trans.columns:
                if col in config.CATGORICAL_FEATURE_LIST:
                    enc_ = LabelEncoder()
                    self.data_trans[col] = enc_.fit_transform(self.data_trans[col])
                elif col in config.NUMERICAL_FEATURE_LIST:
                    scaler = StandardScaler()
                    self.data_trans[col] = scaler.fit_transform(self.data_trans[col].values.reshape(-1,1))

        else:
            self.idx_true, self.idx_false = self.get_encode_idx()
            self.user_idx = self.user_idx[self.user_idx.index.isin(self.idx_true)]
            self.data_trans = self.data_trans[self.data_trans.index.isin(self.idx_true)]
            for col in self.data_trans.columns:
                if col in self.label_encode:
                    cat = self.label_encode[col]
                    enc = LabelEncoder()
                    enc.fit(cat)
                    self.data_trans[col] = enc.transform(self.data_trans[col])

                elif col in self.scale:
                    min_max = self.scale[col]
                    scale_min = min_max['min']
                    scale_max = min_max['max']

                    scaler_ = StandardScaler()
                    scaler_.fit([[scale_min], [scale_max]])
                    self.data_trans[[col]] = scaler_.transform(self.data_trans[[col]])
            self.data_trans = pd.concat([self.user_idx, self.data_trans],axis=1)
        #games transform
        data_ga_group = self.data_games.groupby("trans_id")
        data_games = pd.DataFrame(columns=["trans_id","string"])
        for i in data_ga_group["trans_id"]:
            group = data_ga_group.get_group(i[0])
            group = group[group.columns[~group.columns.isin(config.POP_FEATURE_LIST)]]
            group_string = group[group.columns].astype(str).apply(lambda x: ','.join(x), axis = 1)
            string=''
            for ind in group_string.index:
                string = string + group_string[ind]+' '
                if ind== group_string.index[-1]:
                    string = string + group_string[ind]
            vector = self.ftmodel.get_sentence_vector(string)
            d = {'trans_id': [i[0]], 'string': [string]}
            his = pd.DataFrame(data=d)
            vector=pd.DataFrame(vector).T
            his = pd.concat([his,vector],axis=1)

            data_games = data_games.append(his)            
        data_games = data_games.reset_index()
        self.data_games = data_games.drop(columns=["index","string"])
        #concate trans and games
        transform_inputs = pd.merge(self.data_trans, self.data_games, on="trans_id")
        transform_inputs = transform_inputs.drop(columns="trans_id")
        transform_inputs = transform_inputs[transform_inputs.index.isin(self.idx_true)]
        transform_inputs_to_tensor=tf.convert_to_tensor(transform_inputs)
        
        return transform_inputs_to_tensor
