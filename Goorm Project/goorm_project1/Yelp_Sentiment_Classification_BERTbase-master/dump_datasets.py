import pickle


###################################################################
'''
make dump dataset files with pickle
< mk_dataset >
input: X
output: train_pos, train_neg, dev_pos, dev_neg
'''
###################################################################


def mk_dataset():
    def read_data(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            x = [line.lower() for line in f.readlines()]

        return x
    
    PATH = './datasets/'
    train_pos = read_data(PATH + 'sentiment.train.1')
    train_neg = read_data(PATH + 'sentiment.train.0')

    dev_pos = read_data(PATH + 'sentiment.dev.1')
    dev_neg = read_data(PATH + 'sentiment.dev.0')

    with open("./dump_datasets/train_dev_dumps.p","wb") as f:
        pickle.dump(train_pos, f)
        pickle.dump(train_neg, f)
        pickle.dump(dev_pos, f)
        pickle.dump(dev_neg, f)

    return train_pos, train_neg, dev_pos, dev_neg

def mk_dataset_xlnet():
    def read_data(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            x = [line for line in f.readlines()]

        return x
    
    PATH = './datasets/'
    train_pos = read_data(PATH + 'sentiment.train.1')
    train_neg = read_data(PATH + 'sentiment.train.0')

    dev_pos = read_data(PATH + 'sentiment.dev.1')
    dev_neg = read_data(PATH + 'sentiment.dev.0')

    with open("./dump_datasets/train_dev_dumps_xlnet.p","wb") as f:
        pickle.dump(train_pos, f)
        pickle.dump(train_neg, f)
        pickle.dump(dev_pos, f)
        pickle.dump(dev_neg, f)

    return train_pos, train_neg, dev_pos, dev_neg