from itertools import product
import pickle
import torch
import pandas as pd
import random
import os

from .utils.checkGenerate import caculate_features
from .utils.checkGenerate import auto_encoding_umap
from .utils.checkGenerate import calculate_distance
from .src.vae import VAE


def merge_strings_on_overlap(str_list, n):
    merged_strings = []
    for i, str1 in enumerate(str_list):
        for j, str2 in enumerate(str_list):
            if i != j and str1[-n:] == str2[:n]:
                merged_str = str1 + str2[n:]
                if merged_str not in merged_strings:
                    merged_strings.append(merged_str)
    return merged_strings


def trim_strings(str_list):
    trimmed_str_list = []
    for s in str_list:
        if len(s) > 15:
            # Generate a random number between 0 and len(s) - 16
            rand_num = random.randint(0, len(s) - 16)
            # Randomly choose to trim from the start or the end
            if random.choice([True, False]):
                # Trim from the start
                trimmed_str = s[rand_num:]
            else:
                # Trim from the end
                trimmed_str = s[:len(s) - rand_num]
            trimmed_str_list.append(trimmed_str)
        else:
            trimmed_str_list.append(s)
    return trimmed_str_list


def find_possible_one_hot(input_tensor):
    # Initialize a list to store all possible one-hot matrices
    one_hot_positions = []

    # Iterate over each row in the tensor
    for row in input_tensor:
        # Get the indices of non-zero elements in the row
        non_zero_indices = torch.nonzero(row).flatten().tolist()

        # Add the indices to the list
        one_hot_positions.append(non_zero_indices)

    # Generate all possible combinations of one-hot vectors across all rows
    all_combinations = list(product(*one_hot_positions))
    return all_combinations


# 文件路径
current_dir = os.path.dirname(__file__)
VECTOR2AMINO_DICT_PATH = os.path.abspath(os.path.join(current_dir, 'dicts/vector2amino_dict.pkl'))


def generate(cluster, epoch=20, overLap=3, nomalize_threshold=0.95):
    CLUSTER = ["0", "polar", "pos_neg", "P", "G", "pos", "aliphatic", "neg", "Q"]
    if cluster not in CLUSTER:
        raise ValueError("Invalid cluster name. Please choose from: ['0', 'polar', 'pos_neg', 'P', 'G', 'pos', 'aliphatic', 'neg', 'Q']")

    MODEL_PATH = os.path.abspath(os.path.join(current_dir, f'model_save/vae/{cluster}.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAE()
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    vae.eval()

    fake_input_str_list = []
    for times in range(epoch):
        # 生成随机潜在变量
        z = torch.randn(1, 64, 15)
        # 使用解码器生成新的数据
        reconstructed_img = vae.decoder(z)
        reconstructed_img = reconstructed_img.view(15, 21)
        # Find the maximum value in each row
        max_values, _ = torch.max(reconstructed_img, dim=1, keepdim=True)
        # Divide each row by its maximum value
        normalized_tensor = reconstructed_img / max_values
        # Keep values greater than 0.9 as is, others set to 0
        thresholded_tensor = torch.where(normalized_tensor > nomalize_threshold, normalized_tensor,
                                         torch.zeros_like(normalized_tensor))
        all_combinations = find_possible_one_hot(thresholded_tensor)

        with open(VECTOR2AMINO_DICT_PATH, 'rb') as f:
            vector2amino_dict = pickle.load(f)

        # 通过位点转换为氨基酸
        for combination in all_combinations:
            fake_input_str = ''
            for i in list(combination):
                fake_input_str += vector2amino_dict[i]
            fake_input_str_list.append(fake_input_str)

    fake_input_str_merge = []
    # Example usage
    for i in range(overLap, 15, 1):
        fake_input_str_merge += merge_strings_on_overlap(fake_input_str_list, n=i)

    fake_input_str_list = fake_input_str_list + fake_input_str_merge
    fake_input_str_list = trim_strings(fake_input_str_list)

    # 保留符合分类的
    features = caculate_features(fake_input_str_list)
    umap_result = auto_encoding_umap(features)
    label_result = calculate_distance(umap_result)
    label_result_cluster = list(label_result['huge_cluster'])
    label_result_cluster = [CLUSTER[int(i)] for i in label_result_cluster]
    fake_input_str_list = [fake_input_str_list[i] for i in range(len(label_result_cluster)) if
                           label_result_cluster[i] == cluster]

    result_df = pd.DataFrame(fake_input_str_list, columns=['generate_seqs'])

    os.makedirs('PM_generate', exist_ok=True)
    # check if the file exists, if not, create the file, if it exists, append the data
    if not os.path.exists(f'PM_generate/generate_{cluster}.csv'):
        result_df.to_csv(f'PM_generate/generate_{cluster}.csv', index=False)
        print(f'PM_generate/generate_{cluster}.csv has been created.')
    else:
        result_df.to_csv(f'PM_generate/generate_{cluster}.csv', mode='a', header=False, index=False)
        print(f'PM_generate/generate_{cluster}.csv has been appended.')

    return result_df



