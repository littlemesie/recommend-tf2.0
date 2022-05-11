# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/31 下午3:44
@summary:
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from model import DIN
from ctr.utils.data_process import create_amazon_electronic_dataset

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
maxlen = 10

embed_dim = 8
att_hidden_units = 64
ffn_hidden_units = [256, 128, 64]
dnn_dropout = 0.5
att_activation = 'sigmoid'
ffn_activation = 'prelu'

learning_rate = 0.001
batch_size = 32
epochs = 5

user_dense_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_dense_{}'.format(i) for i in range(5)])
user_sparse_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['user_sparse_{}'.format(i) for i in range(3)])
item_dense_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_dense_{}'.format(i) for i in range(5)])
item_sparse_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_sparse_{}'.format(i) for i in range(3)])
behavior_feature_train = None
for ml in range(maxlen):
    tmp = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_sparse_{}_{}'.format(ml, i) for i in range(3)])
    if ml == 0:
        behavior_feature_train = tmp
    else:
        behavior_feature_train = pd.concat([behavior_feature_train, tmp], axis=1)

target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

# valid
user_dense_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_dense_{}'.format(i) for i in range(5)])
user_sparse_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['user_sparse_{}'.format(i) for i in range(3)])
item_dense_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_dense_{}'.format(i) for i in range(5)])
item_sparse_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_sparse_{}'.format(i) for i in range(3)])
behavior_feature_val = None
for ml in range(maxlen):
    tmp = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_sparse_{}_{}'.format(ml, i) for i in range(3)])
    if ml == 0:
        behavior_feature_val = tmp
    else:
        behavior_feature_val = pd.concat([behavior_feature_val, tmp], axis=1)

target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

sparse_feature_dict = {}
user_sparse_feature_index = {}
item_sparse_feature_index = {}
behavior_feature_index = {}

for idx, col in enumerate(user_sparse_feature_train.columns):
    sparse_feature_dict[col] = (user_sparse_feature_train[col].max() + 1, 64)
    user_sparse_feature_index[col] = idx
for idx, col in enumerate(item_sparse_feature_train.columns):
    sparse_feature_dict[col] = (item_sparse_feature_train[col].max() + 1, 64)
    item_sparse_feature_index[col] = idx
for idx, col in enumerate(behavior_feature_train.columns):
    behavior_feature_index[col] = idx

sparse_feature_index = [user_sparse_feature_index, item_sparse_feature_index, behavior_feature_index]
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    din = DIN(sparse_feature_dict, sparse_feature_index, att_hidden_units, ffn_hidden_units, att_activation,
                    ffn_activation, maxlen, dnn_dropout)
    model = din.build_graph()
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    model.run_eagerly = True

model.fit(
    [user_dense_feature_train, user_sparse_feature_train, item_dense_feature_train, item_sparse_feature_train, behavior_feature_train],
    [target_train],
    epochs=epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
    validation_split=0,
    batch_size=batch_size,
)
