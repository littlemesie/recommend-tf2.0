# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/11/12 上午11:47
@summary: ESMM model for CTR and CVR predict task
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from ctr.layers.modules import DNN


class ESMM(Model):
    def __init__(self, cate_feature_columns, cate_feature_dict, hidden_units=[128, 64],  activation='relu',
                 dropout=0., embed_reg=1e-4):
        super(ESMM, self).__init__()
        self.cate_feature_columns = cate_feature_columns
        self.user_cate_feature_dict, self.item_cate_feature_dict = cate_feature_dict
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.embed_layers = {
            'embed_' + k: Embedding(input_dim=v[0],
                         input_length=1,
                         output_dim=v[1],
                         embeddings_initializer='random_uniform',
                         embeddings_regularizer=l2(embed_reg))
            for k, v in self.cate_feature_columns.items()
        }
        self.user_dnn = DNN(hidden_units, activation, dropout)
        self.item_dnn = DNN(hidden_units, activation, dropout)

    def call(self, inputs, **kwargs):
        """"""
        ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input, \
        cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input = inputs

        ctr_pred = self.build_ctr_model(ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                                        ctr_item_cate_input)
        cvr_pred = self.build_cvr_model(cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                                        cvr_item_cate_input)
        ctcvr_pred = tf.multiply(ctr_pred, cvr_pred)
        outputs = [ctr_pred, ctcvr_pred]
        print(outputs)
        return outputs

    def build_ctr_model(self, ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                        ctr_item_cate_input):
        print(self.user_cate_feature_dict)
        user_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](ctr_user_cate_input[:, v[0]])
                                     for k, v in self.user_cate_feature_dict.items()], axis=-1)

        item_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](ctr_item_cate_input[:, v[0]])
                                     for k, v in self.item_cate_feature_dict.items()], axis=-1)

        user_feature = tf.concat([ctr_user_numerical_input, user_embeddings], axis=-1)
        item_feature = tf.concat([ctr_item_numerical_input, item_embeddings], axis=-1)

        user_feature = self.user_dnn(user_feature)
        item_feature = self.item_dnn(item_feature)

        dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(self.dropout)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(self.hidden_units[-1], activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid', name='ctr_output')(dense_feature)
        return pred

    def build_cvr_model(self, cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                        cvr_item_cate_input):
        user_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](cvr_user_cate_input[:, v[0]])
                                     for k, v in self.user_cate_feature_dict.items()], axis=-1)

        item_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](cvr_item_cate_input[:, v[0]])
                                     for k, v in self.item_cate_feature_dict.items()], axis=-1)

        user_feature = tf.concat([cvr_user_numerical_input, user_embeddings], axis=-1)
        item_feature = tf.concat([cvr_item_numerical_input, item_embeddings], axis=-1)

        user_feature = self.user_dnn(user_feature)
        item_feature = self.item_dnn(item_feature)

        dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(self.dropout)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(self.hidden_units[-1], activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid', name='cvr_output')(dense_feature)
        return pred

    def build_graph(self, **kwargs):
        # CTR model input
        ctr_user_numerical_input = layers.Input(shape=(5,))
        ctr_user_cate_input = layers.Input(shape=(5,))
        ctr_item_numerical_input = layers.Input(shape=(5,))
        ctr_item_cate_input = layers.Input(shape=(3,))

        # CVR model input
        cvr_user_numerical_input = layers.Input(shape=(5,))
        cvr_user_cate_input = layers.Input(shape=(5,))
        cvr_item_numerical_input = layers.Input(shape=(5,))
        cvr_item_cate_input = layers.Input(shape=(3,))

        model = Model(
            inputs=[ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input],
            outputs=self.call([ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input]))

        return model
