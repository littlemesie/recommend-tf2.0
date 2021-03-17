import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
from match.layers.modules import DNN, SampledSoftmaxLayer

class YoutubeDNN(Model):

    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=(), num_sampled=5,
                 user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
               l2_reg_embedding=1e-6, dnn_dropout=0, **kwargs):
        super(YoutubeDNN, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns

        self.user_embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for i, feat in enumerate(self.user_sparse_feature_columns)
        }
        self.item_embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=feat['feat_len'],
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(l2_reg_embedding))
            for i, feat in enumerate(self.item_sparse_feature_columns)
        }
        self.user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)


    def call(self, inputs, training=None, mask=None):

        user_sparse_inputs, user_dense_inputs, item_sparse_inputs, item_dense_inputs, labels = inputs
        # user_sparse_inputs, item_sparse_inputs, labels = inputs
        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(i)](user_sparse_inputs[:, i])
                                  for i in range(user_sparse_inputs.shape[1])], axis=-1)

        # user_dnn_input = tf.concat([user_sparse_embed, user_dense_inputs], axis=-1)
        user_dnn_input = user_sparse_embed
        user_dnn_out = self.user_dnn(user_dnn_input)

        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(i)](item_sparse_inputs[:, i])
                                       for i in range(item_sparse_inputs.shape[1])], axis=-1)

        # item_dnn_input = tf.concat([item_sparse_embed, item_dense_inputs], axis=-1)
        item_dnn_input = item_sparse_embed
        item_dnn_out = self.item_dnn(item_dnn_input)
        # print(item_dnn_out)
        print(item_sparse_inputs[2].shape)
        output = SampledSoftmaxLayer(num_classes=1000000, num_sampled=self.num_sampled)(
            [item_dnn_out, user_dnn_out, labels])

        self.user_embed = user_dnn_out
        self.item_embed = item_dnn_out

        return output

    def summary(self, **kwargs):
        user_sparse_inputs = Input(shape=(len(self.user_sparse_feature_columns), ), dtype=tf.float32)
        user_dense_inputs = Input(shape=(len(self.user_dense_feature_columns),), dtype=tf.int32)
        item_sparse_inputs = Input(shape=(len(self.item_sparse_feature_columns), ), dtype=tf.float32)
        item_dense_inputs = Input(shape=(len(self.item_dense_feature_columns),), dtype=tf.int32)
        labels_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_sparse_inputs, user_dense_inputs, item_sparse_inputs, item_dense_inputs, labels_inputs],
                    outputs=self.call([user_sparse_inputs, user_dense_inputs, item_sparse_inputs, item_dense_inputs,
                                       labels_inputs])).summary()

        # Model(inputs=[user_sparse_inputs, item_sparse_inputs, labels_inputs],
        #       outputs=self.call([user_sparse_inputs, item_sparse_inputs,
        #                          labels_inputs])).summary()

def test_model():
    user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    model = YoutubeDNN(user_features, item_features)
    model.summary()

test_model()

