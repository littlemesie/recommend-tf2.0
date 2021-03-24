import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from match.utils.feature_util import sparseFeature

def get_label(row):
    if row >= 3:
        label = 1
    else:
        label = 0

    return label

def create_ml_100k_dataset(embed_dim=16):
    """加载数据"""
    base_path = '/Users/mesie/python/recommend-learning/data/'
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

    train_X = [train[user_features].values.astype('int32'), train[movie_features].values.astype('int32'),
               train['label'].values.astype('int32')]
    train_y = train['label'].values.astype('int32')

    test_X = [test[user_features].values.astype('int32'), test[movie_features].values.astype('int32'),
              test['label'].values.astype('int32')]
    test_y = test['label'].values.astype('int32')
    print(user_feat_cols)

    print(item_feat_cols)
    print(train_X[0].shape)

    return user_feat_cols, item_feat_cols, train_X, train_y, test_X, test_y

create_ml_100k_dataset()
