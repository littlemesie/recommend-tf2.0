# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/6/25 15:18
@summary:
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
from match.layers.modules import DNN

class Dssm(Model):

    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=(), num_sampled=1,
                 user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
               l2_reg_embedding=1e-6, dnn_dropout=0, **kwargs):
        super(Dssm, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns

        self.user_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for feat in self.user_sparse_feature_columns
        }

        self.item_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for feat in self.item_sparse_feature_columns
        }

        self.user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)

    def cosine_similarity(self, tensor1, tensor2):
        """计算cosine similarity"""
        # 把张量拉成矢量，这是我自己的应用需求
        tensor1 = tf.reshape(tensor1, shape=(1, -1))
        tensor2 = tf.reshape(tensor2, shape=(1, -1))
        # 求模长
        tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
        tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
        # 内积
        tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
        # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
        cosin = tf.realdiv(tensor1_tensor2, tensor1_norm * tensor2_norm)

        return cosin

    def call(self, inputs, training=None, mask=None):

        user_sparse_inputs, item_sparse_inputs = inputs
        # print(user_sparse_inputs, item_sparse_inputs)
        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                  for k, v in user_sparse_inputs.items()], axis=-1)

        user_dnn_input = user_sparse_embed
        self.user_dnn_out = self.user_dnn(user_dnn_input)

        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in item_sparse_inputs.items()], axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)

        cosine_score = self.cosine_similarity(self.item_dnn_out, self.user_dnn_out)
        output = tf.reshape(tf.sigmoid(cosine_score), (-1, 1))

        return output

    def summary(self, **kwargs):

        user_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        # print(self.user_sparse_inputs)
        item_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}


        model = Model(inputs=[user_sparse_inputs, item_sparse_inputs],
              outputs=self.call([user_sparse_inputs, item_sparse_inputs]))

        model.__setattr__("user_input", user_sparse_inputs)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)
        return model


# def model_test():
#     user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     model = Dssm(user_features, item_features)
#     model.summary()
#
# model_test()