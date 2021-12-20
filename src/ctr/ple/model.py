# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/12/11 下午4:25
@summary:
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, BatchNormalization, Softmax
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2

class Expert(Layer):
    def __init__(self, hidden_unit, activation='relu', dnn_dropout=0.5):
        super(Expert, self).__init__()
        self.dnn_network = Dense(units=hidden_unit, activation=activation)
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = BatchNormalization()(inputs)
        x = self.dnn_network(x)
        x = self.dropout(x)
        return x

class Gate(Layer):
    def __init__(self, hidden_unit, activation='relu', dnn_dropout=0.5):
        super(Gate, self).__init__()
        self.dnn_network = Dense(units=hidden_unit, activation=activation)
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = BatchNormalization()(inputs)
        x = self.dnn_network(x)
        x = self.dropout(x)
        return x


class GatingNetwork(Layer):

    def __init__(self, num_experts):
        super(GatingNetwork, self).__init__()
        self.num_expers = num_experts

    def call(self, inputs, *args, **kwargs):
        inputs_features = inputs
        # experts
        expert_outputs = [self.expert(inputs_features) for i in range(self.num_experts)]
        expert_outputs = tf.stack(expert_outputs)
        print(expert_outputs)

        gate_outputs = []
        # gate
        gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(inputs_features.shape[1], self.num_experts),
            initializer=initializers.variance_scaling,
            trainable=True
        ) for i in range(self.num_tasks)]

        gate_bias = [self.add_weight(
            name='gate_bias_task_{}'.format(i),
            shape=(self.num_experts,),
            initializer=initializers.zeros,
            trainable=True
        ) for i in range(self.num_tasks)]

        for index, gate_kernel in enumerate(gate_kernels):
            gate_output = tf.matmul(inputs_features, gate_kernel)

            gate_output += gate_bias[index]
            # gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        return gate_outputs, expert_outputs

class Tower(Layer):
    def __init__(self, hidden_unit, activation='relu', dnn_dropout=0.5):
        super(Tower, self).__init__()
        self.dnn_network = Dense(units=hidden_unit, activation=activation)
        self.dropout = Dropout(dnn_dropout)
        self.softmax = Softmax(axis=1)

    def call(self, inputs, **kwargs):
        """"""
        x = BatchNormalization()(inputs)
        x = self.dnn_network(x)
        x = self.dropout(x)
        x =self.softmax(x)

        return x

class PLE(Model):
    def __init__(self, cate_feature_columns, cate_feature_dict, num_experts=6, expert_hidden_unit=32,
                 tower_hidden_unit=8,  activation='relu', num_tasks=2,
                 dropout=0.5, embed_reg=1e-4):
        super(PLE, self).__init__()
        self.cate_feature_columns = cate_feature_columns
        self.user_cate_feature_dict, self.item_cate_feature_dict = cate_feature_dict
        self.num_experts = num_experts
        self.expert_hidden_unit = expert_hidden_unit
        self.tower_hidden_unit = tower_hidden_unit
        self.activation = activation
        self.dropout = dropout
        self.num_tasks = num_tasks
        self.embed_layers = {
            'embed_' + k: Embedding(input_dim=v[0],
                         input_length=1,
                         output_dim=v[1],
                         embeddings_initializer='random_uniform',
                         embeddings_regularizer=l2(embed_reg))
            for k, v in self.cate_feature_columns.items()
        }

        self.expert = Expert(self.expert_hidden_unit, self.activation, self.dropout)
        self.tower = Tower(self.tower_hidden_unit, self.activation, self.dropout)
        self.gate_network = GatingNetwork(self.num_experts)

    def call(self, inputs, **kwargs):
        """"""
        ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input, \
        cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input = inputs

        user_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](ctr_user_cate_input[:, v[0]])
                                     for k, v in self.user_cate_feature_dict.items()], axis=-1)

        item_embeddings = tf.concat([self.embed_layers['embed_{}'.format(k)](ctr_item_cate_input[:, v[0]])
                                     for k, v in self.item_cate_feature_dict.items()], axis=-1)

        user_feature = tf.concat([ctr_item_numerical_input, user_embeddings], axis=-1)
        item_feature = tf.concat([ctr_item_numerical_input, item_embeddings], axis=-1)
        inputs_features = layers.concatenate([user_feature, item_feature], axis=-1)
        # shared, task1, task2
        shared_gate_outputs, shared_expert_outputs = self.gate_network(inputs_features)
        task1_gate_outputs, task1_expert_outputs = self.gate_network(inputs_features)
        task2_gate_outputs, task2_expert_outputs = self.gate_network(inputs_features)
        # output
        final_outputs = []
        for i, gate_output in enumerate(shared_gate_outputs):
            shared_weighted_expert_output = tf.matmul(shared_expert_outputs, gate_output)
            task1_weighted_expert_output = tf.matmul(task1_expert_outputs, task1_gate_outputs[i])
            task2_weighted_expert_output = tf.matmul(task2_expert_outputs, task2_gate_outputs[i])
            weighted_expert_output = tf.matmul(shared_weighted_expert_output, task1_weighted_expert_output)
            weighted_expert_output = tf.matmul(weighted_expert_output, task2_weighted_expert_output)
            tower_output = self.tower(weighted_expert_output)
            final_outputs.append(tower_output)

        print(final_outputs)
        return final_outputs

    def build_graph(self, **kwargs):
        # CTR model input
        ctr_user_numerical_input = layers.Input(shape=(5,))
        ctr_user_cate_input = layers.Input(shape=(5,))
        ctr_item_numerical_input = layers.Input(shape=(5,))
        ctr_item_cate_input = layers.Input(shape=(3,))

        # CVR model input
        cvr_user_numerical_input = layers.Input(shape=(5,))
        cvr_user_cate_input = layers.Input(shape=(5,))
        cvr_item_numerical_input = layers.Input(shape=(5,))
        cvr_item_cate_input = layers.Input(shape=(3,))

        model = Model(
            inputs=[ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input],
            outputs=self.call([ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input]))

        return model
