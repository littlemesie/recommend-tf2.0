import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Concatenate, Activation, GlobalAveragePooling2D, Reshape
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import Zeros


class DNN(Layer):
    """DNN Layer"""
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        """
        DNN part
        :param hidden_units: A list. List of hidden layer units's numbers
        :param activation: A string. Activation function
        :param dnn_dropout: A scalar. dropout number
        """
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class SampledSoftmaxLayer(Layer):
    """Sampled Softmax Layer"""
    def __init__(self, num_sampled=5, **kwargs):
        super(SampledSoftmaxLayer, self).__init__(**kwargs)
        self.num_sampled = num_sampled

    def build(self, input_shape):
        self.size = input_shape[0][2]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        item_embeddings, user_embeddings, label_idx = inputs_with_label_idx
        item_embeddings = tf.squeeze(item_embeddings, axis=1)  # (None, len)
        # item_embeddings = tf.transpose(item_embeddings)
        user_embeddings = tf.squeeze(user_embeddings, axis=1)  # (None, len)

        loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=user_embeddings,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

def split_heads(x, seq_len, num_heads, depth):
    """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    Args:
        :param x: A Tensor with shape of [batch_size, seq_len, num_heads * depth]
        :param seq_len: A scalar(int).
        :param num_heads: A scalar(int).
        :param depth: A scalar(int).
    :return: A tensor with shape of [batch_size, num_heads, seq_len, depth]
    """
    x = tf.reshape(x, (-1, seq_len, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

def scaled_dot_product_attention(q, k, v, mask):
    """Attention Mechanism Function.
    Args:
        :param q: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param k: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param v: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param mask: A 3d/4d tensor with shape of (None, ..., seq_len, 1)
    :return:
    """
    mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
    # Scaled
    dk = tf.cast(k.shape[-1], dtype=tf.float32)
    scaled_att_logits = mat_qk / tf.sqrt(dk)

    paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)  # (None, seq_len, seq_len)
    outputs = tf.where(tf.equal(mask, tf.zeros_like(mask)), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
    # softmax
    outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len, seq_len)
    outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)

    return outputs

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        """Multi Head Attention Mechanism.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :return:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)


    def call(self, q, k, v, mask):
        # q, k, v, mask = inputs
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)
        # split d_model into num_heads * depth
        seq_len, d_model = q.shape[1], q.shape[2]
        q = split_heads(q, seq_len, self.num_heads, q.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        k = split_heads(k, seq_len, self.num_heads, k.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        v = split_heads(v, seq_len, self.num_heads, v.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        # mask
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_heads, 1, 1])  # (None, num_heads, seq_len, 1)
        # attention
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)  # (None, num_heads, seq_len, d_model // num_heads)
        # reshape
        outputs = tf.reshape(tf.transpose(scaled_attention, [0, 2, 1, 3]), [-1, seq_len, d_model])  # (None, seq_len, d_model)
        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """Feed Forward Network.
        Args:
            :param hidden_unit: A scalar.
            :param d_model: A scalar.
        :return:
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., layer_norm_eps=1e-6):
        """Encoder Layer.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads.
            :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
            :param dropout: A scalar. Number of dropout.
            :param layer_norm_eps: A scalar. Small float added to variance to avoid dividing by zero.
        :return:
        """
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # (None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)  # (None, seq_len, d_model)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2

class PoolingLayer(Layer):
    """"""
    def __init__(self, mode='mean', **kwargs):

        if mode not in ['mean', 'max', 'sum']:
            raise ValueError("mode must be max or mean")
        self.mode = mode
        super(PoolingLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) == 1:
            return inputs[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), inputs))

        a = tf.keras.layers.Concatenate(axis=-1)(expand_seq_value_len_list)
        if self.mode == "mean":
            output = tf.reduce_mean(a, axis=-1)
        elif self.mode == "sum":
            output = tf.reduce_sum(a, axis=-1)
        else:
            output = tf.reduce_max(a, axis=-1)

        return output


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def squash(self, inputs):
        vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
        vec_squashed = scalar_factor * inputs
        return vec_squashed

    def build(self, input_shape):

        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embeddings, seq_len = inputs
        batch_size = tf.shape(behavior_embeddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)

            behavior_embdding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = self.squash(Z)
            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keepdims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = tf.reduce_sum(keys * query, axis=-1, keepdims=True)
        weight = tf.pow(weight, self.pow_p)  # [x,k_max,1]

        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.math.log1p(tf.cast(inputs[2], dtype="float32")) / tf.math.log(2.)  # hist_len
                )
            ), dtype="int64")
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)

        weight = tf.nn.softmax(weight, name="weight")
        output = tf.reduce_sum(keys * weight, axis=1)

        return output


class SELayer(Layer):
    def __init__(self, filter_sq=16, **kwargs):
        # filter_sq 是 Excitation 中第一个卷积过程中卷积核的个数
        super(SELayer, self).__init__(**kwargs)
        self.filter_sq = filter_sq
        self.ave_pool = GlobalAveragePooling1D()
        self.dense = Dense(filter_sq)
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')


    def call(self, inputs, training=None, **kwargs):
        squeeze = self.ave_pool(inputs)

        excitation = self.dense(squeeze)
        excitation = self.relu(excitation)
        excitation = Dense(inputs.shape[-1])(excitation)
        excitation = self.sigmoid(excitation)
        excitation = Reshape((1, 1, inputs.shape[-1]))(excitation)

        output = inputs * excitation

        return output

# import numpy as np
# SE = SELayer(16)
# inputs = np.zeros((1, 32, 32), dtype=np.float32)
# print(SE(inputs).shape)

