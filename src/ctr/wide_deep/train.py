
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from ctr.wide_deep.model import WideDeep
from ctr.utils.data_process import create_criteo_dataset

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '/Users/mesie/python/recommend-learning/data/criteo/sample_100000.txt'
    read_part = True
    sample_num = 100000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        wdl = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
        model = wdl.build_graph()
        # ============================Compile============================
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
