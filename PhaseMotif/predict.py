import torch
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt

from .src.model import PredictMain
from .src.model import AnalyseMain
from .src.guided_backpro import GuidedBackprop
from .utils.seqTrans import seq2Matrix
from .utils.checkGenerate import caculate_features
from .utils.checkGenerate import auto_encoding_umap
from .utils.checkGenerate import calculate_distance

# 文件路径
current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(current_dir, 'model_save/8.pth'))
AMINO = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def cluster(seq_list):
    features = caculate_features(seq_list)
    umap_result = auto_encoding_umap(features)
    label_result = calculate_distance(umap_result)
    label_result = label_result['huge_cluster'].tolist()
    label = ["0", "polar", "pos_neg", "P", "G", "pos", "aliphatic", "neg", "Q"]
    label_result = [label[int(i)] for i in label_result]

    return label_result


def pic(idr_name, idr, density, times, choose_result):
    """Draw the result of the IDR"""
    """
    :param idr_name: str, the IDR name
    :param idr: str, the IDR sequence
    :param density: list, the density of each position
    :param choose_result: list, the number of times each position is selected
    :param times: list, the density of each position being selected
    :return: None
    """
    all_list = list(range(len(idr)))
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(int(len(all_list) / 9), 3),
                                   gridspec_kw={'height_ratios': [1, 1]})
    # 画折线图
    ax1.plot(all_list, density, marker='o')
    ax1.set_xticks(all_list)
    ax1.set_xticklabels(list(idr))
    ax1.set_ylabel('Density', fontsize=15)
    ax1.set_title(idr_name, fontsize=18)
    ax1.grid(False)

    # Remove tick lines
    ax1.tick_params(axis='x', length=0)

    # 画垂直线
    for x in choose_result:
        ax1.axvline(x=x, color='r', linestyle='--')

    # 画折线图
    ax2.bar(all_list, times)
    ax2.set_xticks(all_list)
    ax2.set_xticklabels(list(idr))
    ax2.set_ylabel('Times', fontsize=15)
    ax2.invert_yaxis()  # 倒置y轴
    ax2.xaxis.set_label_position('top')  # 将x轴标签位置设置在顶部
    ax2.xaxis.tick_top()  # 将x轴刻度设置在顶部
    ax2.grid(False)

    # Adjust subplot parameters to increase spacing
    plt.subplots_adjust(hspace=0.3)

    plt.savefig(f'PM_analyse/Pic_result/{idr_name}.png')
    plt.close()


def analyse_main(idr_list, idr_name=None, paint=False):
    """
    :param idr_list: str list, the IDR sequence
    :param idr_name: str list, whether to automatically name the idr in the result
    :param paint: bool, whether to draw the result
    :return: 单个位点的密度、序列被选中的次数密度、重要位点的选择、类别的标签
    """
    if idr_name is None:
        idr_name_list = [f'IDR_{index}' for index in range(len(idr_list))]
    else:
        if len(idr_name) != len(idr_list):
            raise ValueError("The lengths of 'idr_name' and 'idr_list' do not match.")
        idr_name_list = idr_name

    os.makedirs('PM_analyse', exist_ok=True)

    if paint:
        os.makedirs('PM_analyse/Pic_result', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_path = MODEL_PATH

    model = AnalyseMain(cnn1out_channel=8, cnn1kernel=15, cnn1stride=1, cnn1padding=(0, 1), num_head=8,
                        head_size=8, value_size=1, num_level=12)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    gdp = GuidedBackprop(model)

    analyse_result_df = []
    for idr, idr_name in zip(idr_list, idr_name_list):
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
        # choose_result = [list(set(choose_result)).count(i) for i in all_list]

        if paint:
            pic(idr_name, idr, density, times, choose_result)

        seq = seq.split('_')
        seq = [i for i in seq if i != '']
        cluster_label = cluster(seq)

        analyse_result_df.append([idr_name, idr, density, choose_result, times, cluster_label, seq])

    analyse_result_df = pd.DataFrame(analyse_result_df, columns=['IDR Name', 'IDR', 'Density', 'Choose_result', 'Times', 'Cluster_label', 'Key Region'])
    save_df = analyse_result_df.loc[:, ['IDR Name', 'IDR', 'Key Region', 'Cluster_label']]
    save_df = save_df.explode(['Key Region', 'Cluster_label']).reset_index(drop=True)
    save_df.to_csv('PM_analyse/PM_analyse_result.csv', index=False)

    return analyse_result_df


def predict_main(idr_list, idr_name=None):
    """Predict the result of the IDR"""
    """
    :param idr_list: str list, the IDR sequence
    :param idr_name: str list, whether to automatically name the idr in the result
    :return: predict_result, float, the predict result
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = PredictMain(cnn1out_channel=8, cnn1kernel=15, cnn1stride=1, cnn1padding=(0, 1), num_head=8, head_size=8, value_size=1, num_level=12)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs('PM_analyse', exist_ok=True)

    if idr_name is None:
        idr_name_list = [f'IDR_{index}' for index in range(len(idr_list))]
    else:
        if len(idr_name) != len(idr_list):
            raise ValueError("The lengths of 'idr_name' and 'idr_list' do not match.")
        idr_name_list = idr_name

    predict_result_list = []
    for idr, idr_name in zip(idr_list, idr_name_list):
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

        predict_result_list.append([idr_name, idr, predict_result])

    predict_result_list = pd.DataFrame(predict_result_list, columns=['IDR Name', 'IDR', 'Predict Score'])
    predict_result_list.to_csv('PM_analyse/PM_predict_result.csv', index=False)

    return predict_result_list




    




