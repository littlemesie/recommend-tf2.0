# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/31 下午3:47
@summary:
"""

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense, Dropout, ReLU

class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: A list. Neural network hidden units.
        :param dim_stack: A scalar. The dimension of inputs unit.
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs

class FM(Layer):
    """
    Wide part
    """
    def __init__(self, feature_length, w_reg=1e-6):
        """
        Factorization Machine
        In DeepFM, only the first order feature and second order feature intersect are included.
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        # first order
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class CrossNetwork(Layer):
    def __init__(self, layer_num, reg_w=1e-6, reg_b=1e-6):
        """CrossNetwork
        :param layer_num: A scalar. The depth of cross network
        :param reg_w: A scalar. The regularizer of w
        :param reg_b: A scalar. The regularizer of b
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0  # (None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, dim, dim)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l

class DNN(Layer):
    """
    Deep part
    """
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """
        DNN part
        :param hidden_units: A list like `[unit1, unit2,...,]`. List of hidden layer units's numbers
        :param activation: A string. Activation function.
        :param dnn_dropout: A scalar. dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        """
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs, **kwargs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x, **kwargs):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x
