import numpy as np
import math
from collections import Counter
import pickle
import os

# explain_afterbgp(all_gradient)
# explain_Spp(before_SPPData, need_index, gradient_dic, n_large)  # n_large并没有用
# explain_Attention(attention_score, attention_value, need_index, n_large1, n_large2)
# important_position2animo(important_position, x)
# 文件路径
current_dir = os.path.dirname(__file__)
VECTOR2AMINO_DICT_PATH = os.path.join(current_dir, '../dicts/vector2amino_dict.pkl')

# ==================================== explain_afterbgp ========================================= #
def explain_afterbgp(all_gradient):
    """
    :param all_gradient: 得到了第二次卷积前所有值的反向梯度，(8,78)的numpy
    :return: 所有非负的list[(1,10),(7,76)]
    """
    # 获取所有非负值的索引
    indices = np.where(all_gradient > 0)
    positions = list(zip(indices[0], indices[1]))
    values = all_gradient[indices]

    dic = {}
    for x, y, value in zip(indices[0], indices[1], values):
        dic[(x, y)] = value

    return positions, dic


# ==================================   explain_Spp   =================================== #
# 写金字塔池化的返回代码

def gussian(level):  # 判断处于哪个num_level
    return int((level + 1) * level / 2)


# 0 - 1 | 1 2 - 2 | 3 4 5 - 3
def judge_level_group(shu):  # 索引
    i = 1
    while gussian(i) <= shu:
        i += 1
    level = i
    group = shu - gussian(i - 1)
    return level, group  # 返回一个spp数据是从哪个num_level的哪组得到的


# 判断每个值是attention里求最大值的哪部分
def findProbablyPosition_beforeSPP(spp_index, data_len):  # 根据spp池化得到的结果的index和原数据的长度反推每个值是卷积网络里哪部分的最大值
    level, group = judge_level_group(spp_index)
    kernel_size = math.ceil(data_len / level)
    stride = math.ceil(data_len / level)
    pooling = math.floor((kernel_size * level - data_len + 1) / 2)

    if pooling > 0.5 * kernel_size:
        stride = kernel_size - 1
        kernel_size = data_len + stride - level * stride
        pooling = 0

    if pooling != 0 and group == 0:
        result = list(range(0, group * stride - pooling + kernel_size))
    else:
        result = list(range(group * stride - pooling, group * stride - pooling + kernel_size))
    return result


def findMaxIndex(index_list, data_sort_index):  # 找到索引列表中最大值的索引
    for index in data_sort_index[::-1]:
        if index in index_list:
            return index
        else:
            continue
    return None


def explain_Spp(before_SPPData, need_index, gradient_dic, n_large):  # 因为返回代码是一个一个返回的 所以不需要痛苦于数据添0 直接用就行
    """
    :param before_SPPData: 还没有进行spp池化的数据，【8，1，len】 8个卷积核，一维数据
    :param need_index: 需要返回的索引号，（x,y）,x是对应哪个原数据，y是在原数据的位置
    :param n_large: 要几个最大值，即spp计算中被选中的最多次数的值
    :return: 记载位置的list，[[]*8]，其中每一个[]对应一个muti-head中的一头
    """
    result_important_position = [[] for _ in range(8)]  # 每个卷积核卷出的数据的确定区域要分开存放
    important_position_dic = {}
    for which_spp, index in need_index:
        before_SPPData_use = before_SPPData[which_spp].flatten()  # 选择用哪个卷积核卷出的数据
        # print('使用的卷积数据为:', before_SPPData_use)
        # print('SPP位点为：', index)
        probably_important_position = findProbablyPosition_beforeSPP(index, before_SPPData_use.shape[-1])  # 确定可能区域
        # print('可能区域为:', probably_important_position)
        important_position = findMaxIndex(probably_important_position,
                                          np.argsort(before_SPPData_use))  # 确定是某个important_position
        # print('确定区域为:', important_position)
        result_important_position[which_spp].append(important_position)  # 提取important_position到对应行所存储的位置
        try:
            important_position_dic[(which_spp, important_position)].append((which_spp, index))  # 将提取important_position和原位点进行字典保留，方便后续计算
        except:
            important_position_dic[(which_spp, important_position)] = [(which_spp, index)]

    # print('所有important_position:', result_important_position)
    top_important_position = [[] for _ in range(8)]
    for ii in range(8):
        every_spp_pos = list(set(result_important_position[ii]))
        every_hang_gradient = [sum([gradient_dic[each_orign] for each_orign in important_position_dic[(ii, pos)]]) for pos in every_spp_pos]
        # print('every_hang_gradient', every_hang_gradient)
        n_large = int(len(every_hang_gradient))
        indices = sorted(range(len(every_hang_gradient)), key=lambda i: every_hang_gradient[i], reverse=True)[:n_large]
        # print('indices',indices)
        top_important_position[ii] = [every_spp_pos[ind] for ind in indices]
        # print('top_important_position[ii]',top_important_position[ii])

    # print('n_large筛选后的important_position:', top_important_position)
    return top_important_position


# =================================   explain_Attention   ========================================= #
# 写attention层的返回函数
def find_largest_n_indices(input_list, n):  # 朴素地找n个最大值的索引
    indices = []
    for _ in range(n):
        max_index = input_list.index(max(input_list))
        indices.append(max_index)
        input_list[max_index] = float('-inf')  # 将找到的最大值替换为负无穷，避免找到重复的最大值
    return indices


def top_n_with_ties(counter, n): # 获取前n个元素时有概率后半段一样大
    # 使用Counter的most_common方法获取前n个最常见的元素
    top_n = counter.most_common(n)

    # 如果n为0或者counter为空，直接返回空列表
    if not top_n:
        return []

    # 获取排名第n的元素的计数
    nth_count = top_n[-1][1]

    # 获取所有计数与排名第n的元素计数相同的元素
    all_with_nth_count = [item for item in counter.items() if item[1] == nth_count]

    # 将前n-1个元素和所有计数与排名第n的元素计数相同的元素合并
    combined = top_n[:-1] + all_with_nth_count

    new_combined = []
    # 如果返回的元素数量大于n的3倍，只返回那些计数比排名第n的元素的计数大的元素
    if len(combined) > 3 * n:
        new_combined = [item for item in combined if item[1] > nth_count]

    # 但是如果返回的元素数量小于n/2，那还是返回所有元素，多返回一些
    if new_combined == [] or len(new_combined) < n/2:
        return [item[0] for item in combined]

    # 只返回值，不返回计数
    return [item[0] for item in new_combined]


def explain_Attention(attention_score, attention_value, need_index, n_large1, n_large2):
    """
    :param attention_score: 注意力得分是8*n*n的，n是氨基酸的个数
    :param need_index: 需要返回的索引号，[[]*8]，其中每一个[]对应一层attention_score
    :param n_large1: 注意力得分矩阵中和一个位点关键的几个位点
    :param n_large2: 最终返回的位点个数
    :return: 记载位置的list
    """
    # print(attention_value.shape, attention_score.shape)
    result = []
    for iii in range(8):
        if not need_index[iii]:
            continue
        else:
            result = result + need_index[iii] # 先把原位点加进去
            use_attention_score = attention_score[iii]
            use_attention_value = attention_value[iii]
            use_attention_value_beyond0_indices = np.where(use_attention_value>0)
            use_attention_value_beyond0_indices = list(use_attention_value_beyond0_indices[0])
            # print('使用的注意力value:', use_attention_value)
            # print('大于0的位点：', list(use_attention_value_beyond0_indices[0]))
            # print('使用的注意力得分:', use_attention_score)
            for dot in need_index[iii]:
                search_line = use_attention_score[dot]
                # print(f'和{dot}相关的注意力值是：', search_line)
                # indices = find_largest_n_indices(search_line, n_large)
                indices = np.argsort(search_line)[-n_large1:]
                # indices = np.argsort(search_line)
                indices = list(indices)
                indices_beyond0 = [i for i in indices if i in set(use_attention_value_beyond0_indices) ]
                # print('quchuzazhi',indices_beyond0)
                result = result + indices_beyond0  # 和原位点相关的n_large个位点也加进去
                # print(f'和{dot}相关的位点是：', indices)

    counter = Counter(result)
    # print('counter',counter)
    if n_large2 == -1:
        all_times = [item[1] for item in counter.items()]
        all_position = [item[0] for item in counter.items()]
        return all_position, all_times

    top_result = top_n_with_ties(counter, n_large2)
    # print('最终结果：', top_result)
    return top_result


# ================================   explain_firstCNN   ================================== #
# 写第一次卷积的返回代码
def explain_firstCNN_feature(CNN_padding, CNN_Stride, CNN_kernel, need_index, index_times, dataRealLen):
    result_important_position = list()
    start_position = []
    for index, times in zip(need_index, index_times):
        temp = list(range(index * CNN_Stride - CNN_padding, index * CNN_Stride - CNN_padding + CNN_kernel))
        result_important_position = result_important_position + temp * times
        start_temp = [temp[0]] * times
        start_position += start_temp

    result_important_position = [x for x in result_important_position if 0 <= x <= dataRealLen - 1]
    start_position = [x for x in start_position if x >= 0]
    # print(result_important_position)
    return result_important_position, start_position


def explain_firstCNN(CNN_padding, CNN_Stride, CNN_kernel, need_index, dataRealLen):
    result_important_position = set()
    for index in need_index:
        temp = list(range(index * CNN_Stride - CNN_padding, index * CNN_Stride - CNN_padding + CNN_kernel))
        result_important_position = result_important_position | set(temp)

    result_important_position = list(result_important_position)
    result_important_position = [x for x in result_important_position if 0 <= x <= dataRealLen - 1]
    # print(result_important_position)
    return result_important_position


# ==================================   important_position2animo   ================================= #
# 矩阵转氨基酸
def matrix2seq(array, dic):
    findlabels = np.argmax(array)
    return dic[findlabels]


def important_position2animo(important_position, x):
    f_read = open(VECTOR2AMINO_DICT_PATH, 'rb')
    vector2amino_dict = pickle.load(f_read)
    f_read.close()

    seq = ''
    last = -1
    for animo in sorted(important_position):
        if animo == 0:
            seq += matrix2seq(x[:, animo].reshape(-1), vector2amino_dict)
        else:
            if (animo - last - 1) == 0:
                seq = seq + matrix2seq(x[:, animo].reshape(-1), vector2amino_dict)
            else:
                seq = seq + '_' + matrix2seq(x[:, animo].reshape(-1), vector2amino_dict)
        last = animo

    return seq































