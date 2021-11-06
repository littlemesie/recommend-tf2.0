# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/3 下午7:13
@summary: Deep FM model
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
from ctr.layers.modules import FM
from ctr.layers.modules import DNN


class DeepFM(Model):
	def __init__(self, feature_columns, hidden_units=(128, 64, 32), dnn_dropout=0.,
				 activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
		"""
		DeepFM
		:param feature_columns: A list. sparse column feature information.
		:param hidden_units: A list. A list of dnn hidden units.
		:param dnn_dropout: A scalar. Dropout of dnn.
		:param activation: A string. Activation function of dnn.
		:param fm_w_reg: A scalar. The regularizer of w in fm.
		:param embed_reg: A scalar. The regularizer of embedding.
		"""
		super(DeepFM, self).__init__()
		self.dense_feature_columns, self.sparse_feature_columns = feature_columns
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
										 input_length=1,
										 output_dim=feat['embed_dim'],
										 embeddings_initializer='random_normal',
										 embeddings_regularizer=l2(embed_reg))
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.feature_length = 0
		for feat in self.dense_feature_columns:
			self.feature_length += 1
		for feat in self.sparse_feature_columns:
			self.feature_length += feat['embed_dim']

		self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
		self.fm = FM(self.feature_length, fm_w_reg)
		self.dnn = DNN(hidden_units, activation, dnn_dropout)
		self.dense = Dense(1, activation=None)

	def call(self, inputs, **kwargs):
		dense_inputs, sparse_inputs = inputs
		# embedding
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
								  for i in range(sparse_inputs.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)

		embeds = tf.concat([dense_inputs, sparse_embed], axis=-1)
		# fm
		# second_inputs = tf.reshape(sparse_embed, shape=(-1, sparse_embed.shape[1], self.embed_dim))
		fm_outputs = self.fm([embeds, sparse_embed])  # (batch_size, 1)
		# deep
		deep_outputs = self.dnn(embeds)
		deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
		# outputs
		outputs = tf.nn.sigmoid(tf.add(fm_outputs, deep_outputs))
		return outputs

	def build_graph(self, **kwarg):
		dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
		sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
		model = Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs]))
		return model
