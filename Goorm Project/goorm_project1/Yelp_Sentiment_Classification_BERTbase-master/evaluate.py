import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

# from dump_models import load_model
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer
)

from data_processing import regular


###################################################################
'''
evaluate new data
< test_model >
input: model, tokenizer, mean_val_acc, mean_val_loss, file_name='test_no_label', device='cuda'
output: make submission
'''
###################################################################


def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)

    attention_mask = torch.tensor([[1] * len(input_id) + [0] * (max_len - len(input_id)) for input_id in input_ids])
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids], batch_first=True)
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])

    return input_ids, attention_mask, token_type_ids, position_ids

def test_model(model, tokenizer, mean_val_acc, mean_val_loss, file_name='test_no_label', device='cuda'):
    test_df = pd.read_csv('./datasets/' + file_name + '.csv')
    test_df_ = test_df['Id']

    test = [sent.lower() for sent in test_df_]
    # test = [sent for sent in test_df_]


    # test = regular(test)

    test_dataset = [np.array(tokenizer.encode(line)) for line in test]


    test_batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=False, 
                                            collate_fn=collate_fn_style_test,
                                            num_workers=2)
    # for idx, i in enumerate(test_loader):
    #     print(i)
    #     if idx == 1:
    #         break

    model.eval()
    with torch.no_grad():
        predictions = []
        for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader, desc='Test', position=1, leave=None):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)

            output = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids)
            
            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            predictions += batch_predictions
    print(predictions)

    test_df['Category'] = predictions
    test_df.to_csv('./submissions/sub' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)) + '.csv', index=False)

    return predictions


if __name__ == '__main__':
    MODEL_NAME = 'bert-base-uncased'
    # try:
    #     with open('./dump_models_toknizer/' + MODEL_NAME + '.p', 'rb') as f:
    #         print('model exist => just load model')
    #         model = pickle.load(f)
    #         tokenizer = pickle.load(f)
    # except:
    #     print('exeption occur => download model')
    #     model, tokenizer = load_model(MODEL_NAME)

    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    MODEL_NAME = './best_models/model9850'
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # test_model(model, tokenizer, 0, 0, file_name='test_no_label', device='cuda')
    predictions = test_model(model, tokenizer, int(MODEL_NAME[-4:]), int(MODEL_NAME[-4:]), file_name='test_no_label', device='cuda')