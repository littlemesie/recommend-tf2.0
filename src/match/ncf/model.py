import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input

from match.layers.modules import DNN


class NCF(Model):
    def __init__(self, user_feature_columns, item_feature_columns, hidden_units=[64, 16, 8], dropout=0.2,
                 activation='relu', neg_num=10, embed_reg=1e-6, **kwargs):
        """
        NCF model
        :param user_feature_columns: A dict.
        :param item_feature_columns: A dict.
        :param hidden_units: A list.
        :param dropout: A scalar.
        :param activation: A string.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NCF, self).__init__(**kwargs)
        # Now only user_id, item_id
        self.neg_num = neg_num
        # user embedding
        self.user_embedding = Embedding(input_dim=user_feature_columns['feat_num'],
                                           input_length=1,
                                           output_dim=user_feature_columns['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=item_feature_columns['feat_num'],
                                           input_length=1,
                                           output_dim=item_feature_columns['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))

        # neg item embedding
        self.neg_item_embedding = Embedding(input_dim=item_feature_columns['feat_num'],
                                            input_length=self.neg_num,
                                            output_dim=item_feature_columns['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # dnn
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1), (None, 1/101)
        # GMF part
        gmf_user_embed = self.user_embedding(user_inputs)  # (None, 1, dim)
        gmf_pos_embed = self.item_embedding(pos_inputs)  # (None, 1, dim)
        gmf_neg_embed = self.neg_item_embedding(neg_inputs)  # (None, 1/101, dim)
        gmf_pos_vector = tf.nn.sigmoid(tf.multiply(gmf_user_embed, gmf_pos_embed))  # (None, 1, dim)
        gmf_neg_vector = tf.nn.sigmoid(tf.multiply(gmf_user_embed, gmf_neg_embed))  # (None, 1, dim)

        # MLP part
        mlp_user_embed = self.user_embedding(user_inputs)  # (None, 1, dim)
        mlp_pos_embed = self.item_embedding(pos_inputs)  # (None, 1, dim)
        mlp_neg_embed = self.neg_item_embedding(neg_inputs)  # (None, 1/101, dim)

        mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)  # (None, 1, 2 * dim)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)  # (None, 1/101, 2 * dim)
        mlp_pos_vector = self.dnn(mlp_pos_vector)  # (None, 1, dim)
        mlp_neg_vector = self.dnn(mlp_neg_vector)  # (None, 1/101, dim)

        # concat
        pos_vector = tf.concat([gmf_pos_vector, mlp_pos_vector], axis=-1)  # (None, 1, 2 * dim)
        neg_vector = tf.concat([gmf_neg_vector, mlp_neg_vector], axis=-1)  # (None, 1/101, 2 * dim)

        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def build_graph(self, **kwargs):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(self.neg_num,), dtype=tf.int32)
        model = Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs]))
        print(model.outputs)
        return model


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    ncf = NCF(user_features, item_features)
    model = ncf.build_graph()
    model.summary()

test_model()
