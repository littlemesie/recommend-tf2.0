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
from match.layers.modules import EncoderLayer


class SASRec(tf.keras.Model):
    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns,
                 user_dense_feature_columns=(), item_dense_feature_columns=(),
                 user_var_sparse_feature_columns=(), item_var_sparse_feature_columns=(),
                 blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=20, norm_training=True, causality=False, embed_reg=1e-6):
        """
        SASRec model
        """
        super(SASRec, self).__init__()
        # sequence length
        self.maxlen = maxlen
        # user item feature columns
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns
        self.user_var_sparse_feature_columns = user_var_sparse_feature_columns
        self.item_var_sparse_feature_columns = item_var_sparse_feature_columns

        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # d_model must be the same as embedding_dim, because of residual connection
        self.d_model = self.embed_dim

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
        self.dropout = Dropout(dropout)
        # attention block
        self.encoder_layer = [EncoderLayer(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]

    def call(self, inputs, **kwargs):
        # inputs
        seq_inputs, pos_inputs, neg_inputs = inputs  # (None, maxlen), (None, 1), (None, 1)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # pos encoding
        # pos_encoding = positional_encoding(seq_inputs, self.embed_dim)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed  # (None, maxlen, dim)
        att_outputs *= mask

        # self-attention
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        user_info = tf.expand_dims(att_outputs[:, -1], axis=1)  # (None, 1, dim)
        # item info
        pos_info = self.item_embedding(pos_inputs)  # (None, 1, dim)
        neg_info = self.item_embedding(neg_inputs)  # (None, 1/100, dim)
        pos_logits = tf.reduce_sum(user_info * pos_info, axis=-1)  # (None, 1)
        neg_logits = tf.reduce_sum(user_info * neg_info, axis=-1)  # (None, 1)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self,  **kwargs):
        user_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        item_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}

        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)

        model = Model(inputs=[seq_inputs, user_sparse_inputs, item_sparse_inputs],
                      outputs=self.call([seq_inputs, user_sparse_inputs, item_sparse_inputs]))

        model.__setattr__("user_sparse_input", user_sparse_inputs)
        model.__setattr__("item_sparse_input", item_sparse_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)
        return model



def model_test():
    item_fea_col = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    model = SASRec(item_fea_col, num_heads=8)
    m = model.summary()
    m.summary()


model_test()