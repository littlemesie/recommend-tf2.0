import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_ml_100k_dataset(file, trans_score=2, embed_dim=8, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="\t", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # print(data_df.head())
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
                return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    user_feat_col = sparseFeature('user_id', user_num, embed_dim)

    item_feat_col = sparseFeature('item_id', item_num, embed_dim)
    print(user_feat_col, item_feat_col)
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']),
               np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']),
             np.array(val_data['neg_id'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']),
              np.array(test_data['neg_id'])]
    print('============Data Preprocess End=============')
    return user_feat_col, item_feat_col, train, val, test

# file = '/Users/mesie/python/recommend-learning/data/ml-100k/u.data'
# create_ml_100k_dataset(file)