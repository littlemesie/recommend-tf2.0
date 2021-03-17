def sparseFeature(feat, feat_num, feat_len=1, embed_dim=4):
    """
    create dictionary for varlen sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param feat_len: while feature is array: feat_len > 1
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'feat_len': feat_len, 'embed_dim': embed_dim}

def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}

def varLenSparseFeat(feat, feat_num, maxlen, embed_dim=4):
    """
    create dictionary for varlen sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param maxlen: feature array length
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'maxlen': maxlen, 'embed_dim': embed_dim}
