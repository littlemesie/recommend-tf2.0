# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/5/10 下午2:17
@summary: AutoInt 模型
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.regularizers import l2
from ctr.layers.modules import MultiHeadAttention



class AutoInt(Model):
    def __init__(self, feature_columns, att_hidden_units, att_activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        """
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param att_hidden_units: A list. Neural network hidden units.
        :param att_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(AutoInt, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        # attention layer
        self.attention_layer = MultiHeadAttention(head_size=att_hidden_units, activation=att_activation)

        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        # attention
        attention_out = self.attention_layer(x)
        attention_out = tf.reshape(attention_out, shape=[-1, attention_out.shape[1] * attention_out.shape[2]])
        # out
        outputs = self.final_dense(attention_out)
        outputs = tf.nn.sigmoid(outputs)
        return outputs

    def build_graph(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        model = Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs]))
        return model