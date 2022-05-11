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
from ctr.layers.util import split_heads, scaled_dot_product_attention

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
    def __init__(self,  feature_length, w_reg=1e-6):
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
        first_inputs, second_inputs = inputs
        # first order, No w0
        first_order = tf.reduce_sum(tf.matmul(first_inputs, self.w))  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1, keepdims=False)  # (batch_size, 1)
        output = first_order + second_order
        output = tf.reshape(output, shape=(-1, 1))
        return output

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
        x = BatchNormalization()(x)
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class AttentionLayer(Layer):
    def __init__(self, hidden_unit, activation='prelu'):
        """
        """
        super(AttentionLayer, self).__init__()
        self.att_dense = Dense(hidden_unit, activation=activation)

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
        outputs = self.att_dense(info)  # (None, seq_len, 1)

        outputs = tf.reshape(outputs, shape=(-1, outputs.shape[1]))  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        if isinstance(mask, tf.Tensor):
            outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)
        else:
            outputs = paddings

        # outputs = paddings
        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs

class MultiHeadAttention(Layer):

  def __init__(self, head_size, head_num=1, l2_reg=l2(1e-4), activation='relu', use_res=False, name=''):
    """Initializes a `MultiHeadAttention` Layer.

    Args:
      head_num: The number of heads
      head_size: The dimension of a head
      l2_reg: l2 regularizer
      activation: A string. Activation function.
      use_res: Whether to use residual connections before output.
      name: scope of the MultiHeadAttention, so that the parameters could be separated from other MultiHeadAttention
    """
    super(MultiHeadAttention, self).__init__()
    self._head_num = head_num
    self._head_size = head_size
    self._l2_reg = l2_reg
    self._activation = activation
    self._use_res = use_res
    self._name = name

  def _split_multihead_qkv(self, q, k, v):
    """Split multiple heads.

    Args:
      q: Query matrix of shape [bs, feature_num, head_num * head_size].
      k: Key matrix of shape [bs, feature_num, head_num * head_size].
      v: Value matrix of shape [bs, feature_num, head_num * head_size].

    Returns:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].
    """
    reshaped_q = tf.reshape(
        q, shape=[-1, q.shape[1], self._head_num, self._head_size])
    q = tf.transpose(reshaped_q, perm=[0, 2, 1, 3])
    reshaped_k = tf.reshape(
        k, shape=[-1, k.shape[1], self._head_num, self._head_size])
    k = tf.transpose(reshaped_k, perm=[0, 2, 1, 3])
    reshaped_v = tf.reshape(
        v, shape=[-1, v.shape[1], self._head_num, self._head_size])
    v = tf.transpose(reshaped_v, perm=[0, 2, 1, 3])
    return q, k, v

  def _scaled_dot_product_attention(self, q, k, v):
    """Calculate scaled dot product attention by q, k and v.

    Args:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].

    Returns:
      q: Query matrix of shape [bs, head_num, feature_num, head_size].
      k: Key matrix of shape [bs, head_num, feature_num, head_size].
      v: Value matrix of shape [bs, head_num, feature_num, head_size].
    """
    product = tf.linalg.matmul(
        a=q, b=k, transpose_b=True) / (
            self._head_size**-0.5)
    weights = tf.nn.softmax(product)
    out = tf.linalg.matmul(weights, v)
    return out

  def _compute_qkv(self, q, k, v):
    """Calculate q, k and v matrices.

    Args:
      q: Query matrix of shape [bs, feature_num, d_model].
      k: Key matrix of shape [bs, feature_num, d_model].
      v: Value matrix of shape [bs, feature_num, d_model].

    Returns:
      q: Query matrix of shape [bs, feature_num, head_size * n_head].
      k: Key matrix of shape [bs, feature_num, head_size * n_head].
      v: Value matrix of shape [bs, feature_num, head_size * n_head].
    """
    q = Dense(
        units=self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        activation=self._activation)(q)
    k = Dense(
        units=self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        activation=self._activation)(k)
    v = Dense(
        self._head_num * self._head_size,
        use_bias=False,
        kernel_regularizer=self._l2_reg,
        activation=self._activation)(v)
    return q, k, v

  def _combine_heads(self, multi_head_tensor):
    """Combine the results of multiple heads.

    Args:
      multi_head_tensor: Result matrix of shape [bs, head_num, feature_num, head_size].

    Returns:
      out: Result matrix of shape [bs, feature_num, head_num * head_size].
    """
    x = tf.transpose(multi_head_tensor, perm=[0, 2, 1, 3])
    out = tf.reshape(x, shape=[-1, x.shape[1], x.shape[2] * x.shape[3]])
    return out

  def call(self, inputs, **kwargs):
    """Build multiple heads attention layer.

    Args:
      attention_input: The input of interacting layer, has a shape of [bs, feature_num, d_model].

    Returns:
      out: The output of multi head attention layer, has a shape of [bs, feature_num, head_num * head_size].
    """
    if isinstance(inputs, list):
      assert len(inputs) == 3 or len(inputs) == 1, \
          'If the input of multi_head_attention is a list, the length must be 1 or 3.'

      if len(inputs) == 3:
        ori_q = inputs[0]
        ori_k = inputs[1]
        ori_v = inputs[2]
      else:
        ori_q = inputs[0]
        ori_k = inputs[0]
        ori_v = inputs[0]
    else:
      ori_q = inputs
      ori_k = inputs
      ori_v = inputs

    q, k, v = self._compute_qkv(ori_q, ori_k, ori_v)
    q, k, v = self._split_multihead_qkv(q, k, v)
    multi_head_tensor = self._scaled_dot_product_attention(q, k, v)

    out = self._combine_heads(multi_head_tensor)
    if self._use_res:
      W_0_x = Dense(
          out.shape[2],
          use_bias=False,
          kernel_regularizer=self._l2_reg,
          activation=self._activation)(ori_v)
      res_out = tf.nn.relu(out + W_0_x)
      return res_out
    else:
      return out

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x, **kwargs):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x
