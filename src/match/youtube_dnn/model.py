import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
from match.layers.modules import DNN, SampledSoftmaxLayer

class YoutubeDNN(Model):

    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=(), num_sampled=1,
                 user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
               l2_reg_embedding=1e-6, dnn_dropout=0, **kwargs):
        super(YoutubeDNN, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns

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

        self.user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.sampled_softmax = SampledSoftmaxLayer(num_sampled=self.num_sampled)


    def call(self, inputs, training=None, mask=None):

        user_sparse_inputs, item_sparse_inputs, labels = inputs

        user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                  for k, v in user_sparse_inputs.items()], axis=-1)

        user_dnn_input = user_sparse_embed
        self.user_dnn_out = self.user_dnn(user_dnn_input)

        item_sparse_embed = tf.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                       for k, v in item_sparse_inputs.items()], axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)


        output = self.sampled_softmax([self.item_dnn_out, self.user_dnn_out, labels])

        return output

    def summary(self, **kwargs):

        user_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.user_sparse_feature_columns}
        item_sparse_inputs = {uf['feat']: Input(shape=(1, ), dtype=tf.float32) for uf in
                              self.item_sparse_feature_columns}

        labels_inputs = Input(shape=(1,), dtype=tf.int32)

        model = Model(inputs=[user_sparse_inputs, item_sparse_inputs, labels_inputs],
              outputs=self.call([user_sparse_inputs, item_sparse_inputs,
                                 labels_inputs]))
        model.__setattr__("user_input", user_sparse_inputs)
        model.__setattr__("item_input", item_sparse_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)

        model.summary()
        return model


# def test_model():
#     user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     model = YoutubeDNN(user_features, item_features)
#     model.summary()
#
# test_model()

