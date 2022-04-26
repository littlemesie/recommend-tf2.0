# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/4/26 下午3:35
@summary: https://arxiv.org/abs/1906.00091
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.regularizers import l2
from ctr.layers.modules import DNN


class DLRM(Model):
    def __init__(self, feature_columns, bot_dnn_hidden_units=[64, 32, 16], top_dnn_hidden_units=[128, 64],
                 activation='relu', dnn_dropout=0., embed_reg=1e-4):
        """
         DLRM
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param bot_dnn_hidden_units: A list. Bot Neural network hidden units.
        :param top_dnn_hidden_units: A list. Top Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DLRM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.bot_dnn = DNN(bot_dnn_hidden_units, activation, dnn_dropout)
        self.top_dnn = DNN(top_dnn_hidden_units, activation, dnn_dropout)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        dense_fea = self.bot_dnn(self.dense_inputs)
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)

        x = tf.concat([sparse_embed, dense_fea], axis=-1)
        # top_dnn
        top_dnn = self.dnn_network(x)
        top_dnn = self.final_dense(top_dnn)
        # out
        outputs = tf.nn.sigmoid(top_dnn)
        return outputs

    def build_graph(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        model = Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs]))

        return model


