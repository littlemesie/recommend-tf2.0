# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/22 19:16
@summary:
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Embedding, Input
from match.layers.modules import TransformerEncoder


class SASRec(Model):
    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns,
                 user_dense_feature_columns=(), item_dense_feature_columns=(),
                 blocks=1, num_heads=1, att_hidden_unit=128, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, seq_len=10, neg_len=100, embed_reg=1e-6):
        """
        SASRec model
        """
        super(SASRec, self).__init__()
        # sequence length
        self.seq_len = seq_len
        self.neg_len = neg_len
        # user item feature columns
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns

        # d_model must be the same as seq_embed_dim, because of residual connection
        self.d_model = att_hidden_unit

        self.user_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                                    input_length=feat['feat_len'],
                                                    output_dim=feat['embed_dim'],
                                                    embeddings_initializer='random_uniform',
                                                    embeddings_regularizer=l2(embed_reg))
            for feat in self.user_sparse_feature_columns
        }

        self.item_embed_layers = {
            'embed_' + str(feat['feat']): Embedding(input_dim=feat['feat_num'],
                                                    input_length=feat['feat_len'],
                                                    output_dim=feat['embed_dim'],
                                                    embeddings_initializer='random_uniform',
                                                    embeddings_regularizer=l2(embed_reg))
            for feat in self.item_sparse_feature_columns
        }

        self.dropout = Dropout(dnn_dropout)
        # attention block
        self.encoder_layer = [TransformerEncoder(self.d_model, num_heads, ffn_hidden_unit,
                                                 dnn_dropout, layer_norm_eps) for _ in range(blocks)]

    def call(self, inputs, **kwargs):
        # inputs
        seq_inputs, pos_inputs, neg_inputs = inputs
        # # user embed
        # user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
        #                                for k, v in user_sparse_inputs.items()], axis=-1)
        # print(user_sparse_embed)
        # # item embed
        # item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
        #                                for k, v in item_sparse_inputs.items()], axis=-1)

        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)

        # seq embed 没加positional embed
        seq_embed = self.user_embed_layers['embed_seq_item'](seq_inputs)
        # pos embed
        pos_embed = self.user_embed_layers['embed_pos_item'](pos_inputs)
        # neg embed
        neg_embed = self.user_embed_layers['embed_neg_item'](neg_inputs)

        att_outputs = seq_embed  # (None, seq_len, embed_dim)
        att_outputs *= mask
        # self-attention
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        seq_info = tf.expand_dims(att_outputs[:, -1], axis=1)  # (None, 1, dim)
        self.embed = seq_info
        pos_scores = tf.reduce_sum(tf.multiply(seq_info, pos_embed), axis=-1)  # (None, 1)
        neg_scores = tf.reduce_sum(seq_info * neg_embed, axis=-1)  # (None, 1)
        # "binary entropy loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def build_graph(self, **kwargs):
        user_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        item_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}

        seq_inputs = Input(shape=(self.seq_len,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(self.neg_len,), dtype=tf.int32)

        model = Model(inputs=[seq_inputs, pos_inputs, neg_inputs],
                      outputs=self.call([seq_inputs, pos_inputs, neg_inputs]))

        user_input = user_sparse_inputs.update({'seq_item': seq_inputs})

        model.__setattr__("user_input", user_input)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("embed", self.embed)
        return model



# def model_test():
#     user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8},
#                      {'feat': 'seq_item', 'feat_num': 100, 'feat_len': 10, 'embed_dim': 64},
#                      {'feat': 'pos_item', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 64},
#                      {'feat': 'neg_item', 'feat_num': 100, 'feat_len': 100, 'embed_dim': 64}]
#     item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 32}]
#     model = SASRec(user_features, item_features, att_hidden_unit=64)
#     m = model.build_graph()
#     m.summary()
#
#
# model_test()