import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

class FM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        Factorization Machines
        :param feature_columns: a list containing dense and sparse column feature information
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.feature_length = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
                              + len(self.dense_feature_columns)
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

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # one-hot encoding
        sparse_inputs = tf.concat(
            [tf.one_hot(sparse_inputs[:, i],
                        depth=self.sparse_feature_columns[i]['feat_num'])
             for i in range(sparse_inputs.shape[1])
             ], axis=1)
        stack = tf.concat([dense_inputs, sparse_inputs], axis=1)
        # first order
        first_order = self.w0 + tf.matmul(stack, self.w)
        # second order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(stack, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)

        outputs = first_order + second_order
        out = tf.nn.sigmoid(outputs)

        return out

    def build_graph(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        model = Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs]))
        return model
