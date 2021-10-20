import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from match.youtube_dnn.model import YoutubeDNN
from match.utils.data_process import create_ml_100k_dataset
from match.utils.loss_util import sampledsoftmaxloss
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    # ========================= Hyper Parameters =======================
    embed_dim = 16
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 128
    epochs = 10

    # ========================== Create dataset =======================
    user_feat_cols, item_feat_cols, train_X, train_y, test_X, test_y = create_ml_100k_dataset(embed_dim=embed_dim)

    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        youtube = YoutubeDNN(user_sparse_feature_columns=user_feat_cols, item_sparse_feature_columns=item_feat_cols, dnn_dropout=dnn_dropout)
        model = youtube.summary()
        # ============================Compile============================
        model.compile(loss=sampledsoftmaxloss, optimizer=Adam(learning_rate=learning_rate))

    # ============================model checkpoint======================
    # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================
    train_X = [train_X[0], train_X[1], train_y]
    print(train_X)
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )

    # user_embs, item_embed = model.predict(
    #     train_X,
    #     batch_size=batch_size
    # )
    # print(item_embed)
    # print(item_embed.shape)
    # ===========================Test==============================
    # print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
