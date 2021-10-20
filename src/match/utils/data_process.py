import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from match.utils.feature_util import sparseFeature


def get_label(row):
    if row >= 3:
        label = 1
    else:
        label = 0

    return label

def create_ml_100k_dataset(embed_dim=16):
    """加载数据"""
    base_path = '/Users/mesie/Pycharm/recommend/recommend-learning/data/'
    rating_df = pd.read_csv(base_path + 'ml-100k/u.data', sep='\t',
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])

    user_df = pd.read_csv(base_path + 'ml-100k/u.user', sep='|',
                            names=['user_id', 'age', 'gender', 'occupation', 'zip'])

    item_df = pd.read_csv(base_path + 'ml-100k/u.item', sep='|', encoding="ISO-8859-1",
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'url', 'unknown',
                                   'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                   'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                                   'Sci-Fi', 'Thriller', 'War', 'Western'])

    data_df = pd.merge(rating_df, user_df, how="left")
    data_df = pd.merge(data_df, item_df, how="left")
    data_df['label'] = data_df['rating'].apply(get_label)

    # 处理age
    data_df['age'] = pd.cut(data_df['age'], bins=[0, 15, 25, 35, 45, 60, 100], labels=['0-15', '15-25', '25-35',
                                                                                       '35-45', '45-60', '60-100'])
    # 处理电影类型
    user_features = ['user_id', 'age', 'gender']
    movie_features = ['movie_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                      'Sci-Fi', 'Thriller', 'War', 'Western']

    # features = user_features + movie_features
    feature_max_idx = {}
    for feature in user_features + movie_features:
        lbe = LabelEncoder()
        data_df[feature] = lbe.fit_transform(data_df[feature])
        feature_max_idx[feature] = data_df[feature].max() + 1


    user_feat_cols = []
    user_feat_cols.append(sparseFeature(feat='user_id', feat_num=feature_max_idx['user_id'], embed_dim=embed_dim))
    user_feat_cols = user_feat_cols + [sparseFeature(feat=uf, feat_num=feature_max_idx[uf]) for uf in ['age', 'gender']]

    item_feat_cols = []
    item_feat_cols.append(sparseFeature(feat='movie_id', feat_num=feature_max_idx['movie_id'], embed_dim=embed_dim))
    movie_type = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                      'Sci-Fi', 'Thriller', 'War', 'Western']
    item_feat_cols = item_feat_cols + [sparseFeature(feat=mt, feat_num=feature_max_idx[mt]) for mt in movie_type]

    train, test = train_test_split(data_df, test_size=0.2)
    train_X = [{feat: train[feat].values for feat in user_features},
               {feat: train[feat].values for feat in movie_features}]
    # train_X = [train[user_features].values.astype('int32'), train[movie_features].values.astype('int32'),
    #            train['label'].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_X = [{feat: test[feat].values for feat in user_features},
              {feat: test[feat].values for feat in movie_features}]

    # test_X = [test[user_features].values.astype('int32'), test[movie_features].values.astype('int32'),
    #           test['label'].values.astype('int32')]
    test_y = test['label'].values.astype('int32')
    print(train_X)
    return user_feat_cols, item_feat_cols, train_X, train_y, test_X, test_y



def create_sasrec_dataset(trans_score=2, embed_dim=8, maxlen=10, test_neg_num=20):
    """
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    base_path = '/Users/mesie/Pycharm/recommend/recommend-learning/data/ml-latest-small/ratings.csv'
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(base_path, names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # # trans score
    # data_df = data_df[data_df.label >= trans_score]
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
                test_data['hist'].append(hist_i)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['hist'].append(hist_i)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['hist'].append(hist_i)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # item feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    item_feat_col = sparseFeature('item_id', item_num, embed_dim)
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    # padding
    print('==================Padding===================')
    train = [pad_sequences(train_data['hist'], maxlen=maxlen), np.array(train_data['pos_id']),
               np.array(train_data['neg_id'])]
    val = [pad_sequences(val_data['hist'], maxlen=maxlen), np.array(val_data['pos_id']),
             np.array(val_data['neg_id'])]
    test = [pad_sequences(test_data['hist'], maxlen=maxlen), np.array(test_data['pos_id']),
             np.array(test_data['neg_id'])]
    print('============Data Preprocess End=============')
    return item_feat_col, train, val, test

# create_ml_100k_dataset()

# item_feat_col, train, val, test = create_sasrec_dataset()
#
# print(item_feat_col)
# print(train)