import pickle

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer
)


###################################################################
'''
make dump model, tokenizer files with pickle
< load_model >
input: model name
output: model, tokenizer    +   create model_name.p
'''
###################################################################


def load_model(model_name):
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    with open('./dump_models_tokenizer/' + model_name + '.p','wb') as f:
        pickle.dump(model, f)
        pickle.dump(tokenizer, f)
    
    return model, tokenizer

def load_model_xlnet(model_name):
    model = XLNetForSequenceClassification.from_pretrained(model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    with open('./dump_models_tokenizer/' + model_name + '.p','wb') as f:
        pickle.dump(model, f)
        pickle.dump(tokenizer, f)
    
    return model, tokenizer