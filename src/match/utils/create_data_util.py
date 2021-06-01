# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/6/1 下午8:00
@summary:
"""
import pandas as pd
import numpy as np
"""
    data include ctr data and cvr data, ctr data include ctr user data and ctr item data,
    user data include numerical data and categorical data
    item data include numerical data and categorical data
    we generate sample data include user feature data and item feature data
    user feature data include 5 numerical data and 5 categorical data
    item feature data include 5 numerical data and 3 categorical data
"""

# train data
ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((100000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(100000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((100000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(100000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])
cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((100000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(100000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((100000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(100000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])

# val data
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

# train data label
ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=100000))
cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=100000))

# val data label
ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))


def generate_ctr_cvr_data():
    """generate ctr cvr data"""
    train_data = [ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
                  ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                  cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train]
    val_data = [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
                ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val]


    cate_feature_dict = {}
    user_cate_feature_dict = {}
    item_cate_feature_dict = {}
    for idx, col in enumerate(ctr_user_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
        user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
    for idx, col in enumerate(ctr_item_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_item_cate_feature_train[col].max() + 1
        item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)

    return train_data, val_data, cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict


def generate_cvr_data():
    """generate  cvr data"""
    train_data = [cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                  cvr_item_numerical_feature_train, cvr_item_cate_feature_train, cvr_target_train]
    val_data = [cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                cvr_item_numerical_feature_val, cvr_item_cate_feature_val, cvr_target_val]

    cate_feature_dict = {}
    user_cate_feature_dict = {}
    item_cate_feature_dict = {}
    for idx, col in enumerate(ctr_user_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
        user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
    for idx, col in enumerate(ctr_item_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_item_cate_feature_train[col].max() + 1
        item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)

    return train_data, val_data, cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict

train_data, val_data, cate_feature_dict, user_cate_feature_dict, user_cate_feature_dict = generate_cvr_data()
print(user_cate_feature_dict)