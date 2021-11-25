# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/31 下午3:43
@summary:
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l2
from ctr.layers.modules import AttentionLayer, Dice


class DIN(Model):
    def __init__(self, sparse_feature_dict, sparse_feature_index, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=10, dnn_dropout=0.,
                 embed_reg=1e-4):
        """
        DIN
        :param sparse_feature_dict: A dict. sparse_feature_columns
        :param sparse_feature_index: A dict.  sparse_feature_index
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dnn_dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DIN, self).__init__()
        self.maxlen = maxlen
        self.sparse_feature_dict = sparse_feature_dict
        self.user_sparse_feature_index, self.item_sparse_feature_index, self.behavior_feature_index = sparse_feature_index
        self.embed_layers = {
            'embed_' + k: Embedding(input_dim=v[0],
                                    input_length=1,
                                    output_dim=v[1],
                                    embeddings_initializer='random_uniform',
                                    embeddings_regularizer=l2(embed_reg))
            for k, v in self.sparse_feature_dict.items()
        }

        # attention layer
        self.attention_layer = AttentionLayer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)
        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.final_output = Dense(1)

    def call(self, inputs, **kwargs):

        user_dense_input, user_sparse_input, item_dense_input, item_sparse_input, behavior_input = inputs
        print(behavior_input)
        print(self.user_sparse_feature_index)
        # user embed
        user_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](user_sparse_input[:, v])
                                     for k, v in self.user_sparse_feature_index.items()], axis=-1)
        user_embed = tf.concat([user_dense_input, user_embeddings], axis=-1)
        # item embed
        item_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](item_sparse_input[:, v])
                                     for k, v in self.item_sparse_feature_index.items()], axis=-1)
        item_embed = tf.concat([item_sparse_input, item_embeddings], axis=-1)

        # attention ---> mask
        mask = tf.concat([behavior_input[:, i*3] for i in range(self.maxlen)], axis=-1)

        # behavior embed
        behavior_embed = tf.concat([self.embed_layers['embed_{}'.format(f"{k.split('_')[0]}_{k.split('_')[1]}_{k.split('_')[3]}")]
                           (behavior_input[:, v]) for k, v in self.behavior_feature_index.items()], axis=-1)

        behavior_embed = tf.reshape(behavior_embed, shape=(-1, self.maxlen, item_embeddings.shape[1]))

        # att
        att_outputs = self.attention_layer([item_embeddings, behavior_embed, behavior_embed, mask])

        all_inputs = tf.concat([user_embed, item_embed, att_outputs], axis=-1)

        all_inputs = self.bn(all_inputs)

        # ffn
        for dense in self.ffn:
            all_inputs = dense(all_inputs)

        all_inputs = self.dropout(all_inputs)
        #
        outputs = tf.nn.sigmoid(self.final_output(all_inputs))
        print(outputs)
        # outputs = Dense(1, activation='sigmoid', name='output')(all_inputs)
        return outputs

    def build_graph(self, **kwargs):
        user_dense_input = Input(shape=(5,))
        user_sparse_input = Input(shape=(3,))
        item_dense_input = Input(shape=(5,))
        item_sparse_input = Input(shape=(3,))
        behavior_input = Input(shape=(3*self.maxlen,))

        model = Model(inputs=[user_dense_input, user_sparse_input, item_dense_input, item_sparse_input, behavior_input],
                       outputs=self.call([user_dense_input, user_sparse_input, item_dense_input, item_sparse_input, behavior_input]))
        return model
