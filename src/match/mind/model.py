# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/10/21 下午4:58
@summary: MIND模型
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Input
from match.layers.modules import DNN, SampledSoftmaxLayer, PoolingLayer, CapsuleLayer, LabelAwareAttention

class MIND(Model):

    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, hist_item_sparse_feature_columns,
                 user_dense_feature_columns=(),item_dense_feature_columns=(), num_sampled=1, hist_len=3, k_max=2, p=1.0,
                 mode='mean', user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
                 dnn_activation='relu', l2_reg_embedding=1e-6, dnn_dropout=0, **kwargs):
        super(MIND, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.hist_len = hist_len
        self.k_max = k_max
        self.p = p
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns
        self.hist_item_sparse_feature_columns = hist_item_sparse_feature_columns
        self.mode = mode

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
        self.sampled_softmax = SampledSoftmaxLayer(num_sampled=self.num_sampled)
        self.pools = PoolingLayer(mode=self.mode)


    def call(self, inputs, training=None, mask=None):

        user_sparse_inputs, item_sparse_inputs, hist_item_sparse_inputs, labels = inputs
        # user embed
        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                  for k, v in user_sparse_inputs.items()], axis=-1)

        # item embed
        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in item_sparse_inputs.items()], axis=-1)
        item_pool_out = self.pools([item_sparse_embed])

        # hist item embed
        hist_item_embed_list = []
        # print(hist_item_sparse_inputs)
        for hist_input in hist_item_sparse_inputs:
            hist_item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                           for k, v in hist_input.items()], axis=-1)
            # hist_item_sparse_embed = tf.squeeze(hist_item_sparse_embed, axis=1)  # (None, len)

            hist_item_embed_list.append(hist_item_sparse_embed)

        hist_item_embed_list = tf.concat(hist_item_embed_list, axis=1)
        hist_item_embed = self.pools([hist_item_embed_list])

        hist_len = user_sparse_inputs['hist_len']
        # Multi Interest Layer
        high_capsule = CapsuleLayer(input_units=hist_item_embed.shape[2],
                                          out_units=hist_item_embed.shape[2],
                                          max_len=self.hist_len,
                                          k_max=self.k_max)((hist_item_embed, hist_len))

        user_sparse_embed = tf.tile(user_sparse_embed, [1, tf.shape(high_capsule)[1], 1])

        user_deep_input = tf.concat([user_sparse_embed, high_capsule], axis=-1)

        user_dnn_out = self.user_dnn(user_deep_input)
        item_dnn_out = self.item_dnn(item_pool_out)
        # label aware attention
        user_embedding_final = LabelAwareAttention(k_max=self.k_max, pow_p=self.p)((user_dnn_out, item_dnn_out))
        user_embedding_final = tf.expand_dims(user_embedding_final, axis=1)

        self.user_embedding = user_embedding_final
        self.item_embedding = item_dnn_out

        output = self.sampled_softmax([item_dnn_out, user_embedding_final, labels])

        return output

    def build_graph(self, **kwargs):

        user_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        item_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}
        hist_item_sparse_inputs = [{uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.hist_item_sparse_feature_columns} for i in range(self.hist_len)]

        labels_inputs = Input(shape=(1,), dtype=tf.int32)

        model = Model(inputs=[user_sparse_inputs, item_sparse_inputs, hist_item_sparse_inputs, labels_inputs],
              outputs=self.call([user_sparse_inputs, item_sparse_inputs,
                                hist_item_sparse_inputs, labels_inputs]))
        model.__setattr__("user_input", user_sparse_inputs)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("user_embeding", self.user_embedding)
        model.__setattr__("item_embeding", self.item_embedding)

        return model


def test_model():
    user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8},
                     {'feat': 'hist_len', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 32}]
    # hist item
    hist_item_sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8},
                                 {'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8},
                                 {'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    mind = MIND(user_features, item_features, hist_item_sparse_features)
    model = mind.build_graph()
    model.summary()

test_model()


