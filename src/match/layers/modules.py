import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Concatenate
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

def scaled_dot_product_attention(q, k, v, mask, causality=True):
    """
    Attention Mechanism
    :param q: A 3d tensor with shape of (None, seq_len, depth), depth = d_model // num_heads
    :param k: A 3d tensor with shape of (None, seq_len, depth)
    :param v: A 3d tensor with shape of (None, seq_len, depth)
    :param mask:
    :param causality: Boolean. If True, using causality, default True
    :return:
    """
    mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
    dk = tf.cast(k.shape[-1], dtype=tf.float32)
    # Scaled
    scaled_att_logits = mat_qk / tf.sqrt(dk)

    paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)

    # Causality
    if causality:
        diag_vals = tf.ones_like(outputs)  # (None, seq_len, seq_len)
        masks = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (None, seq_len, seq_len)

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (None, seq_len, seq_len)

    # softmax
    outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len, seq_len)
    outputs = tf.matmul(outputs, v)  # (None, seq_len, depth)

    return outputs


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, causality=True):
        """
        Multi Head Attention Mechanism
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :param causality: Boolean. If True, using causality, default True

        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causality = causality

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)

        # split d_model into num_heads * depth, and concatenate
        q = tf.reshape(tf.concat([tf.split(q, self.num_heads, axis=2)], axis=0),
                       (-1, q.shape[1], q.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        k = tf.reshape(tf.concat([tf.split(k, self.num_heads, axis=2)], axis=0),
                       (-1, k.shape[1], k.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        v = tf.reshape(tf.concat([tf.split(v, self.num_heads, axis=2)], axis=0),
                       (-1, v.shape[1], v.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)

        # attention
        scaled_attention = scaled_dot_product_attention(q, k, v, mask, self.causality)  # (None * num_heads, seq_len, d_model // num_heads)

        # Reshape
        outputs = tf.concat(tf.split(scaled_attention, self.num_heads, axis=0), axis=2)  # (N, seq_len, d_model)

        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """
        Feed Forward Network
        :param hidden_unit: A scalar. W1
        :param d_model: A scalar. W2
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., norm_training=True, causality=True):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, causality)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=norm_training)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        # self-attention
        att_out = self.mha([x, x, x, mask])  # （None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2

class PoolingLayer(Layer):

    def __init__(self, mode='mean', **kwargs):

        if mode not in ['mean', 'max', 'sum']:
            raise ValueError("mode must be max or mean")
        self.mode = mode
        super(PoolingLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None, **kwargs):
        if self.mode == "mean":
            output = tf.reduce_mean(inputs, axis=-1)
        elif self.mode == "max":
            output = tf.reduce_max(inputs, axis=-1)
        else:
            output = tf.reduce_sum(inputs, axis=-1)
        return output

def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed

class MultiInterestLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(MultiInterestLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(MultiInterestLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embeddings, seq_len = inputs
        batch_size = behavior_embeddings.shape[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = squash(Z)
            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keep_dims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)
        print("111", interest_capsules)
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

# history_emb = tf.keras.Input(shape=(None,))
# print(history_emb)
# high_capsule = MultiInterestLayer(input_units=16,
#                                 out_units=16, max_len=None,
#                                 k_max=2)((history_emb, 10))