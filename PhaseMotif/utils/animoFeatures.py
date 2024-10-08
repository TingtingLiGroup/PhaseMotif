# poly类型的函数
import re
import pandas as pd


def find_poly(str, poly_char, count):
    """
    :param str: 需要寻找的母字符串
    :param poly_char: 有可能poly的氨基酸 如'Q'
    :param count: 多少个连续的氨基酸算poly
    :return: 符合poly的子字符串个数 0就是没有这个poly
    """
    pattern = f'{poly_char}{{{count},}}'  # 'g{5,}'就是超过5个的g {{}}来表示一个普通的{}
    matches = re.findall(pattern, str)
    return len(matches)


# rich类型的函数
def find_rich(str, rich_char_list, window_size_min, rich_percent):
    """
    :param str: 需要寻找的母字符串
    :param rich_char_list: 有可能rich的一类氨基酸 如正电负电啥的
    :param window_size_min: 窗口大小的最小值
    :param rich_percent: 所占比例
    :return: Trur or False 是否rich
    """
    for window_size in range(window_size_min, len(str) + 1):
        for i in range(0, len(str) - window_size + 1):
            if sum(str[i:i + window_size].count(char) for char in rich_char_list) >= rich_percent * window_size:
                return True
    return False


def calculate_protion(str, special_char_list):
    """
    :param str: 母字符串
    :param special_char_list: 需要计算比例的char们
    :return: 比例
    """
    counts = sum([str.count(special_char) for special_char in special_char_list])
    return counts / len(str)


def calculate_double_protion(str, special_char):
    """
    :param str: 母字符串
    :param special_char_list: 需要计算比例的char们
    :return: 专供两字母比例 AG+GA-0.5*(AGA+GAG)
    """
    special_char_reversed = special_char[::-1]
    special_char_overlap1 = special_char + special_char[0]
    special_char_overlap2 = special_char_reversed + special_char[1]
    counts = str.count(special_char) + str.count(special_char_reversed) - 0.5 * (
                str.count(special_char_overlap1) + str.count(special_char_overlap2))

    str = str.replace('_', '')

    return 2 * counts / len(str)


# from multiprocessing import Pool

# Assuming find_rich function is defined elsewhere

# def check_richness(row, rich_all_list, window_size_min=15, rich_percent=0.5):  # 多线程
#     feature_idr = row['feature_idr']
#     temp_result = [row['index'], row['gene'], feature_idr]
#     for rich in rich_all_list:
#         temp_rich = 1 if find_rich(feature_idr, rich, window_size_min, rich_percent) else 0
#         temp_result.append(temp_rich)
#     return temp_result


# def parallel_apply(df, func, rich_all_list, num_processes):  # 多线程
#     with Pool(num_processes) as pool:
#         results = pool.starmap(func, [(row, rich_all_list) for _, row in df.iterrows()])
#     return results


def feature_extract(feature_data, feature_type):
    """
    :param feature_data: 写了提取到的特征的df 至少需要 ['index', 'gene', 'feature_idr']
    :param feature_type: poly? rich? portion? double_portion? 
    :return: df格式文件
    """
    amino_all = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'U']
    result = []
    columns = []
    if feature_type == 'poly':  # poly
        for _, row in feature_data.iterrows():  # 按行遍历
            index = row['index']
            feature_idr = row['feature_idr']
            gene = row['gene']
            columns = ['index', 'gene', 'feature_idr']
            temp_result = [index, gene, feature_idr]
            # 准备给每个小序列测试所有的poly
            for amino in amino_all:
                poly_which = f'poly_{amino}'  # 提取的poly
                columns.append(poly_which)
                temp_poly = find_poly(feature_idr, amino, 6)  # feature中poly的个数
                temp_result.append(temp_poly)

            result.append(temp_result)
        result = pd.DataFrame(result, columns=columns)
        return result

    # ====================== rich和portion需要的准备数据 ====================================
    rich_all_list = [[x] for x in amino_all]  # 需要rich的类型以及类名
    columns_for_all = amino_all  # 所有氨基酸
    rich_by_special = [['R', 'H', 'K', 'D', 'E']]  # 特殊的分类 想起来一个是一个
    columns_for_special = ['pos_neg']
    rich_by_another_classify = [['R', 'H', 'K'], ['D', 'E'], ['S', 'T', 'N', 'Q'], ['A', 'I', 'L', 'M', 'V'],
                                ['F', 'Y', 'W']]  # 欣欣的那个分类还有三类, ['C'], ['G'], ['P']
    columns_for_another_classify = ['positive', 'negative', 'polars', 'aliphatics', 'aromatics']  # 'single-C',
    # 'single-G','single-P'

    rich_all_list += rich_by_special
    columns_for_all += columns_for_special
    rich_all_list += rich_by_another_classify
    columns_for_all += columns_for_another_classify

    # if feature_type == 'rich':  # rich 多线程
    #     num_processes = 4
    #     results = parallel_apply(feature_data, check_richness, rich_all_list, num_processes)
    #     columns = ['index', 'gene', 'feature_idr'] + [f'rich_{x}' for x in columns_for_all]
    #     result = pd.DataFrame(results, columns=columns)
    #     return result

    if feature_type == 'rich':  # rich 单线程
        for _, row in feature_data.iterrows():
            index = row['index']
            feature_idr = row['feature_idr']
            gene = row['gene']
            columns = ['index', 'gene', 'feature_idr']
            temp_result = [index, gene, feature_idr]
            for rich in rich_all_list:
                temp_rich = 1 if find_rich(feature_idr, rich, 15, 0.5) else 0
                temp_result.append(temp_rich)
            result.append(temp_result)
        columns += [f'rich_{x}' for x in columns_for_all]
        result = pd.DataFrame(result, columns=columns)
        return result

    if feature_type == 'portion':  # 各种元素在在feature中占的比例
        for _, row in feature_data.iterrows():  # 按行遍历
            index = row['index']
            feature_idr = row['feature_idr']
            gene = row['gene']
            columns = ['index', 'gene', 'feature_idr']
            temp_result = [index, gene, feature_idr]
            for rich in rich_all_list:
                temp_protion_feature = calculate_protion(feature_idr, rich)
                temp_result.append(temp_protion_feature)
            result.append(temp_result)
        columns += [f'portion_{x}' for x in columns_for_all]
        result = pd.DataFrame(result, columns=columns)
        return result

    # ====================== 双氨基酸占比需要的准备数据 ====================================
    amino_use = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    animo_double = [x + y for x in amino_use for y in amino_use if x < y]
    # 计算双氨基酸占比
    if feature_type == 'double_portion':  # 双氨基酸的比例
        for _, row in feature_data.iterrows():  # 按行遍历
            index = row['index']
            feature_idr = row['feature_idr']
            gene = row['gene']
            columns = ['index', 'gene', 'feature_idr']
            temp_result = [index, gene, feature_idr]
            temp_feature = []
            for animo in animo_double:
                double_portion_feature = calculate_double_protion(feature_idr, animo)
                temp_feature.append(double_portion_feature)

            temp_feature = temp_result + temp_feature
            result.append(temp_feature)
        columns += [f'portion_{xx}' for xx in animo_double]
        result = pd.DataFrame(result, columns=columns)
        return result  # 返回在特征中的双氨基酸占比