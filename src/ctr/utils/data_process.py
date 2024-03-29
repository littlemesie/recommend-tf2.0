
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

random.seed(2020)

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}

def convert_float(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

    if read_part:
        data_df = pd.read_csv(file, iterator=True)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat].astype(str))

    # ==============Feature Engineering===================

    # dense_features = [feat for feat in data_df.columns if feat not in sparse_features + ['label']]


    for feat in dense_features:
        mms = MinMaxScaler()
        data_df[feat] = mms.fit_transform(data_df[dense_features].astype(int))

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]

    train, test = train_test_split(data_df, test_size=test_size)

    train_X = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_X = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)

def to_df(file_path):
    """
    转化为DataFrame结构
    :param file_path: 文件路径
    :return:
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

def create_amazon_electronic_dataset(embed_dim=8, maxlen=40):
    """
    load amazon electronic dataset
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    reviews_df = to_df('../../../data/reviews_Electronics_10.json')
    meta_df = to_df('../../../data/meta_Electronics.json')
    # 只保留reviews文件中出现过的商品
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    meta_df = meta_df[['asin', 'categories']]

    # 类别只保留最后一个
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

    # meta_df文件的物品ID映射
    asin_map, asin_key = build_map(meta_df, 'asin')

    # meta_df文件物品种类映射
    cate_map, cate_key = build_map(meta_df, 'categories')
    # reviews_df文件的用户ID映射
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')

    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]

    # reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)

    # 各个物品对应的类别
    cate_list = np.array(meta_df['categories'], dtype='int32')

    reviews_df.columns = ['user_id', 'item_id', 'time']


    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # test_data.append([hist_i, [pos_list[i]], 1])
                # test_data.append([hist_i, [neg_list[i]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # val_data.append([hist_i, [pos_list[i]], 1])
                # val_data.append([hist_i, [neg_list[i]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # train_data.append([hist_i, [pos_list[i]], 1])
                # train_data.append([hist_i, [neg_list[i]], 0])

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

def create_census_dataset(embed_dim=8, seed=1):
    """
    census data: https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    """
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv('/data/python/data/census/census-income.data.gz', sep=',', names=column_names)
    test_df = pd.read_csv('/data/python/data/census/census-income.test.gz', sep=',', names=column_names)
    print(train_df)
    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    # One-hot encoding categorical columns
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_raw_labels = train_df[label_columns]
    test_raw_labels = test_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_test = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)

    # # Filling the missing column in the other set
    # transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    test_income = to_categorical((test_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    test_marital = to_categorical((test_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_test_labels = {
        'income': test_income,
        'marital': test_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_test.sample(frac=0.5, replace=False, random_state=seed).index
    test_indices = list(set(transformed_test.index) - set(validation_indices))
    validation_data = transformed_test.iloc[validation_indices]
    validation_label = [dict_test_labels[key][validation_indices] for key in sorted(dict_test_labels.keys())]
    test_data = transformed_test.iloc[test_indices]
    test_label = [dict_test_labels[key][test_indices] for key in sorted(dict_test_labels.keys())]
    # train data
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info

# train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = create_census_dataset()
# print(test_label)