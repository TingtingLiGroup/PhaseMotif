from .animoFeatures import feature_extract
from ..src.model import Autoencoder
from ..src.dataset import MyAutoencoderDataset

import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import umap.umap_ as umap
import os

# 文件路径
current_dir = os.path.dirname(__file__)
AUTOENCODER_MODEL_PATH = os.path.abspath(os.path.join(current_dir, '../model_save/autoencoder_model.pth'))
UMAP_MODEL_PATH = os.path.abspath(os.path.join(current_dir, '../model_save/umap_model.joblib'))
MEANINGFUL_FEATURES_PATH = os.path.abspath(os.path.join(current_dir, '../dicts/meaningful_features.txt'))
SMALL_UMAP_CLUSTER_HUGE_PATH = os.path.abspath(os.path.join(current_dir, '../dicts/small_umap_cluster_huge.csv'))

# Load the Autoencoder model
autoencoder = Autoencoder(input_dim=207)
autoencoder.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH))
autoencoder.eval()

# Load the UMAP model
reducer = joblib.load(UMAP_MODEL_PATH)

# 读取需要使用的列
with open(MEANINGFUL_FEATURES_PATH, 'r') as file:
    meaningful_features = [line.strip() for line in file]

# 读取过去的umap分类
existing_umap = pd.read_csv(SMALL_UMAP_CLUSTER_HUGE_PATH, index_col=None)


# 提取特征的函数
def caculate_features(fake_list):
    # 整理数据
    feature_data = pd.DataFrame(fake_list, columns=['feature_idr'])
    feature_data.reset_index(inplace=True)
    feature_data['gene'] = feature_data['index'].apply(lambda x: f'gene{x}')

    # 提取特征 poly
    poly_feature = feature_extract(feature_data, 'poly')
    # rich
    rich_feature = feature_extract(feature_data, 'rich')
    # portion
    portion_feature = feature_extract(feature_data, 'portion')
    # double_portion
    double_portion_feature = feature_extract(feature_data, 'double_portion')

    # Step 1: Concatenate the DataFrames by columns
    concatenated_df = pd.concat([poly_feature, rich_feature, portion_feature, double_portion_feature], axis=1)
    feature_data = concatenated_df[meaningful_features]

    return feature_data


# Autoencoder and UMAP
def auto_encoding_umap(feature_data):
    # autoencoder编码
    dataset = MyAutoencoderDataset(feature_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_encoded = []
    for data in dataloader:
        encoded, _ = autoencoder(data)
        all_encoded.append(encoded.detach().numpy())
    all_encoded = pd.DataFrame(np.array(all_encoded).reshape(-1, 20))

    # 使用UMAP进行降维
    umap_result = reducer.transform(all_encoded)

    return umap_result


# 计算距离
def calculate_distance(umap_result):
    features_array = existing_umap[['Feature 1', 'Feature 2']].values
    labels_array = existing_umap['huge_cluster'].values

    # 计算距离
    # Calculate the Euclidean distance between each point in the embedding_list and all points in the standrad_embedding DataFrame
    distances = np.sqrt(np.sum((umap_result[:, np.newaxis] - features_array) ** 2, axis=2))

    # Find the index of the minimum distance for each point in the embedding_list
    min_distance_indices = np.argmin(distances, axis=1)
    # Use these indices to get the corresponding labels from the standrad_embedding DataFrame
    min_labels = labels_array[min_distance_indices]

    # 按行遍历，如果距离大于1，那么就认为是新的类别
    label_result = []
    for point, min_distance, min_label in zip(umap_result, np.min(distances, axis=1), min_labels):
        if min_distance > 0.8:
            min_label = -1
        label_result.append([point[0], point[1], min_label])
    label_result = pd.DataFrame(label_result, columns=['Feature 1', 'Feature 2', 'huge_cluster'])

    return label_result

















