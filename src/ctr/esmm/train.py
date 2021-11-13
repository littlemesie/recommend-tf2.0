# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/11/12 上午11:47
@summary:
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from ctr.esmm.model import ESMM

# data include ctr data and cvr data, ctr data include ctr user data and ctr item data,
# user data include numerical data and categorical data
# item data include numerical data and categorical data
# we generate sample data include user feature data and item feature data
# user feature data include 5 numerical data and 5 categorical data
# item feature data include 5 numerical data and 5 categorical data

ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])


ctr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)), columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                         columns=['item_cate_{}'.format(i) for i in range(3)])

ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

train_data = [ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
              ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
              cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train]
val_data = [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
            ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
            cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val]

print(ctr_user_cate_feature_train)
# 'user_cate_0': (10, 64), 'user_cate_1': (10, 64), 'user_cate_2': (10, 64), 'user_cate_3': (10, 64)}
cate_feature_dict = {}
user_cate_feature_dict = {}
item_cate_feature_dict = {}
for idx, col in enumerate(ctr_user_cate_feature_train.columns):
    cate_feature_dict[col] = (ctr_user_cate_feature_train[col].max() + 1, 64)
    user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
for idx, col in enumerate(ctr_item_cate_feature_train.columns):
    cate_feature_dict[col] = (ctr_item_cate_feature_train[col].max() + 1, 64)
    item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)
print(cate_feature_dict)
print(user_cate_feature_dict)

embed_dim = 8
dropout = 0.5
hidden_units = [256, 128, 64]

learning_rate = 0.001
batch_size = 128
epochs = 10
opt = Adam(learning_rate=0.003, decay=0.0001)
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    esmm = ESMM(cate_feature_columns=cate_feature_dict, cate_feature_dict=[user_cate_feature_dict, item_cate_feature_dict])
    model = esmm.build_graph()
    model.summary()
    # ============================Compile============================
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"],
                  loss_weights=[1.0, 1.0],
                  optimizer=opt,
                  metrics=[AUC()])
    model.run_eagerly = True
# ============================model checkpoint======================
# check_path = '../save/esmm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
# checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
#                                                 verbose=1, period=5)
# ==============================Fit==============================

model.fit([ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
             ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
             cvr_item_numerical_feature_train, cvr_item_cate_feature_train],
             [ctr_target_train, cvr_target_train],
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(
                 [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
                 ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                 cvr_item_numerical_feature_val,
                 cvr_item_cate_feature_val], [ctr_target_val, cvr_target_val]))