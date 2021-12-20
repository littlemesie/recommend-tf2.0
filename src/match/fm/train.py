# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/12/20 下午4:57
@summary:
"""
import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from match.fm.model import FM
from match.layers.modules import DNN
from match.utils.data_process import create_ml_100k_dataset

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    # ========================= Hyper Parameters =======================
    embed_dim = 16
    hidden_units = [128, 64]
    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # ========================== Create dataset =======================
    user_feat_cols, item_feat_cols, train_X, train_y, test_X, test_y = create_ml_100k_dataset(embed_dim=embed_dim)
    print(train_y, len(train_y))
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        fm = FM(user_sparse_feature_columns=user_feat_cols, item_sparse_feature_columns=item_feat_cols, k=32)
        model = fm.build_graph()
        model.summary()
        # ============================Compile============================
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))

    # ============================model checkpoint======================
    # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )

    user_embed_model = Model(inputs=model.user_input, outputs=model.user_embeds)
    item_embed_model = Model(inputs=model.item_input, outputs=model.item_embeds)

    user_embs = user_embed_model.predict(test_X[0])
    item_embs = item_embed_model.predict(test_X[1])
    user_embs = DNN(hidden_units=hidden_units)(user_embs)
    item_embs = DNN(hidden_units=hidden_units)(item_embs)
    print(user_embs.shape)
    print(item_embs.shape)
    # ===========================recommend==============================
    index = faiss.IndexFlatIP(item_embs.shape[1])
    # faiss.normalize_L2(item_embs)
    index.add(np.array(item_embs))
    # faiss.normalize_L2(user_embs)
    D, I = index.search(np.ascontiguousarray(user_embs), 10)

    item_index_mapping = {}  # {item_matrix_index: item_id}
    index = 0
    for i, item_id in enumerate(train_X[1]['movie_id']):
        item_index_mapping[index] = int(item_id)
        index += 1


    recommed_dict = {}
    for i, uid in enumerate(test_X[0]['user_id']):
        recommed_dict.setdefault(uid, [])
        try:
            pred = [item_index_mapping[x] for x in I[i]]
            recommed_dict[uid] = pred
        except:
            print(i)

    print(recommed_dict)

    # test_user_items = dict()
    # for ts in test_set:
    #     if ts[0] not in test_user_items:
    #         test_user_items[ts[0]] = set(ts[1])
    # item_popularity = dict()
    # for ts in train_set:
    #     for item in ts[1]:
    #         if item in item_popularity:
    #             item_popularity[item] += 1
    #         else:
    #             item_popularity.setdefault(item, 1)
    #
    # precision = metric.precision(recommed_dict, test_user_items)
    # recall = metric.recall(recommed_dict, test_user_items)
    # coverage = metric.coverage(recommed_dict, item_set)
    # popularity = metric.popularity(item_popularity, recommed_dict)
    #
    # print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage,
    #                                                                                    popularity))