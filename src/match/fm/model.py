# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/12/20 下午3:44
@summary: FM模型
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.python.keras.layers import Lambda
from match.utils.util import reduce_sum

class FM(Model):
    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, k, w_reg=1e-4, v_reg=1e-4,
                 l2_reg_embedding=1e-6):
        """
        Factorization Machines
        :param user_sparse_feature_columns: the user column feature
        :param item_sparse_feature_columns: the item column feature
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns

        self.feature_length = sum([feat['embed_dim'] for feat in self.user_sparse_feature_columns]) \
                              + sum([feat['embed_dim'] for feat in self.item_sparse_feature_columns])

        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        # add weight
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, self.feature_length),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

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

    def call(self, inputs, **kwargs):

        user_sparse_inputs, item_sparse_inputs = inputs
        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in user_sparse_inputs.items()], axis=-1)
        self.user_embeds = Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=False))(user_sparse_embed)

        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in item_sparse_inputs.items()], axis=-1)
        self.item_embeds = Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=False))(item_sparse_embed)

        stack = tf.concat([user_sparse_embed, item_sparse_embed], axis=-1)

        stack = tf.reshape(stack, (-1, stack.shape[2]))
        self.embeds = stack
        # first order
        first_order = self.w0 + tf.matmul(stack, self.w)
        # second order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(stack, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
        outputs = first_order + second_order

        out = tf.reshape(tf.sigmoid(outputs), (-1, 1))
        return out

    def build_graph(self, **kwargs):
        user_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        # print(user_sparse_inputs)
        item_sparse_inputs = {uf['feat']: Input(shape=(1,), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}

        model = Model(inputs=[user_sparse_inputs, item_sparse_inputs],
                      outputs=self.call([user_sparse_inputs, item_sparse_inputs]))

        model.__setattr__("user_input", user_sparse_inputs)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("user_embeds", self.user_embeds)
        model.__setattr__("item_embeds", self.item_embeds)
        model.__setattr__("embeds", self.embeds)
        return model

def model_test():
    user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    fm = FM(user_features, item_features, k=64)
    model = fm.build_graph()
    model.summary()

model_test()
