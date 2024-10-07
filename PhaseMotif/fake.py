import torch
from src.model import PredictMain
from utils.seqTrans import seq2Matrix
import pandas as pd
import sys
import json

def predict_main(idr_list):
    """Predict the result of the IDR"""
    """
    :param idr_list: str list, the IDR sequence
    :return: predict_result, float, the predict result
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = PredictMain(cnn1out_channel=8, cnn1kernel=15, cnn1stride=1, cnn1padding=(0, 1), num_head=8, head_size=8, value_size=1, num_level=12)
    model.load_state_dict(torch.load('model_save/8.pth', map_location=device))
    model.to(device)
    model.eval()

    predict_result_list = []
    for idr in idr_list:
        # print(idr)
        data_one_hot = torch.tensor(seq2Matrix(idr, 'onehot')).unsqueeze(0).float()
        data_alphabet = torch.tensor((seq2Matrix(idr, 'alphabet'))).unsqueeze(0).float()
        result = model([data_one_hot], [data_alphabet], device)
        predict_result = torch.sigmoid(result[0]).item()
        predict_result_list.append(predict_result)

    return predict_result_list


if __name__ == '__main__':
    data = json.loads(sys.argv[1])

    start, end = map(int, data['position'].split('-'))
    
    # 获取两个序列
    seq1 = ''.join(data['idr'][:start])
    seq2 = ''.join(data['idr'][end+1:])

    label = {"0":0, "polar":1, "pos_neg":2, "P":3, "G":4, "pos":5, "aliphatic":6, "neg":7, "Q":8}
    cluster = label.get(data['cluster'])

    fake = pd.read_csv(f'fake_data/temp_fake_data_{cluster}.csv', index_col=None)
    fake['fake_new_idr'] = seq1 + fake['fake_idr'].astype(str) + seq2
    fake_score = predict_main(fake['fake_new_idr'].tolist())
    fake_score = pd.DataFrame(fake_score, columns=['fake_score'])
    fake = pd.concat([fake, fake_score], axis=1)

    # Generate CSV file
    csv_file = '/PUBLIC/output.csv'
    save_file = 'PUBLIC/output.csv'
    fake.to_csv(save_file, index=False)

    print(csv_file)
