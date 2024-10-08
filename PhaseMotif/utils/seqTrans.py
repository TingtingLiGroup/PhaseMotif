import numpy as np
import pickle
import os

# 文件路径
current_dir = os.path.dirname(__file__)
AMINO2VECTOR_DICT_PATH = os.path.join(current_dir, '../dicts/amino2Vector_dict.pkl')
AMINO2ALPHA_DICT_PATH = os.path.join(current_dir, '../dicts/amino2Alpha_dict.pkl')


def seq2Matrix(seq, method):           # 输入IDR序列 输出氨基酸矩阵
    """
    :param method: onehot编码or字母表编码alphabet
    :param seq: ACKLIRKSLTACC
    :return: Matrix
    """
    if method == 'onehot':
        f_read = open(AMINO2VECTOR_DICT_PATH, 'rb')
        amino2Vector_dict = pickle.load(f_read)
        f_read.close()
    elif method == 'alphabet':
        f_read = open(AMINO2ALPHA_DICT_PATH, 'rb')
        amino2Vector_dict = pickle.load(f_read)
        f_read.close()
    else:
        print('没有这个方法')
        return
    matrix = None
    for s in seq:
        if matrix is None:
            matrix = amino2Vector_dict[s].reshape(-1, 1)
            continue
        matrix = np.hstack((matrix, amino2Vector_dict.get(s, amino2Vector_dict['U']).reshape(-1, 1)))
    return matrix






