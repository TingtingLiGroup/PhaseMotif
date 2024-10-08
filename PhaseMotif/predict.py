import torch
from .src.model import PredictMain
from .utils.seqTrans import seq2Matrix
import pandas as pd
import os
import sys
import json

from .src.model import AnalyseMain
from .src.guided_backpro import GuidedBackprop

from .utils.checkGenerate import caculate_features
from .utils.checkGenerate import auto_encoding_umap
from .utils.checkGenerate import calculate_distance

# 文件路径
current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(current_dir, 'model_save/8.pth'))
AMINO = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# 混合了原文件中predict、importance、特征提取和聚类分类cluster
def cluster(seq_list):
    features = caculate_features(seq_list)
    umap_result = auto_encoding_umap(features)
    label_result = calculate_distance(umap_result)
    label_result = label_result['huge_cluster'].tolist()
    label = ["0", "polar", "pos_neg", "P", "G", "pos", "aliphatic", "neg", "Q"]
    label_result = [label[int(i)] for i in label_result]

    return label_result


def analyse_main(idr_list):
    """
    :param idr_list: str list, the IDR sequence
    :return: 单个位点的密度、序列被选中的次数密度、重要位点的选择、类别的标签
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_path = MODEL_PATH

    model = AnalyseMain(cnn1out_channel=8, cnn1kernel=15, cnn1stride=1, cnn1padding=(0, 1), num_head=8,
                        head_size=8, value_size=1, num_level=12)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    gdp = GuidedBackprop(model)

    analyse_result_df = []
    for idr in idr_list:
        # 检查idr的长度
        if len(idr) < 50:
            raise ValueError(f"Error: The length of IDR '{idr}' is less than 50.")
        # 检查idr的字符是否属于字符列表animo
        if not all(char in AMINO for char in idr):
            raise ValueError(f"Error: The IDR '{idr}' contains characters not in AMINO.")
        data_one_hot = torch.tensor(seq2Matrix(idr, 'onehot')).unsqueeze(0).float()
        data_alphabet = torch.tensor((seq2Matrix(idr, 'alphabet'))).unsqueeze(0).float()
        label = torch.tensor([1]).float()
        protein = 'test'
        loader = [([data_one_hot], [data_alphabet], [label], [protein])]

        result, position = gdp.visualize(loader=loader, device=device, divide=0,
                                         feature=True)  # result是全部重要位置，position是每15个重要位置的起点
        seq, choose_result = gdp.visualize(loader=loader, device=device, divide=0,
                                           feature=False)  # seq是关键序列，choose_result是关键序列的重要位置
        result = result['Feature'].tolist()[0]
        position = position['Position'].tolist()[0]
        seq = seq['Feature'].tolist()[0]
        choose_result = choose_result['Position'].tolist()[0]

        all_list = list(range(len(idr)))
        density = [result.count(i) / len(result) for i in all_list]
        density = [i / max(density) for i in density]
        times = [position.count(i) / sum(position) for i in all_list]
        times = [i / max(times) for i in times]
        choose_result = [list(set(choose_result)).count(i) for i in all_list]

        seq = seq.split('_')
        seq = [i for i in seq if i != '']
        cluster_label = cluster(seq)

        analyse_result_df.append([idr, density, choose_result, times, cluster_label])

    analyse_result_df = pd.DataFrame(analyse_result_df, columns=['IDR', 'Density', 'Choose_result', 'Times', 'Cluster_label'])
    return analyse_result_df


def predict_main(idr_list):
    """Predict the result of the IDR"""
    """
    :param idr_list: str list, the IDR sequence
    :return: predict_result, float, the predict result
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = PredictMain(cnn1out_channel=8, cnn1kernel=15, cnn1stride=1, cnn1padding=(0, 1), num_head=8, head_size=8, value_size=1, num_level=12)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    predict_result_list = []
    for idr in idr_list:
        # 检查idr的长度
        if len(idr) < 50:
            raise ValueError(f"Error: The length of IDR '{idr}' is less than 50.")
        # 检查idr的字符是否属于字符列表animo
        if not all(char in AMINO for char in idr):
            raise ValueError(f"Error: The IDR '{idr}' contains characters not in AMINO.")
        data_one_hot = torch.tensor(seq2Matrix(idr, 'onehot')).unsqueeze(0).float()
        data_alphabet = torch.tensor((seq2Matrix(idr, 'alphabet'))).unsqueeze(0).float()
        result = model([data_one_hot], [data_alphabet], device)
        predict_result = torch.sigmoid(result[0]).item()
        predict_result_list.append(predict_result)

    return predict_result_list




    




