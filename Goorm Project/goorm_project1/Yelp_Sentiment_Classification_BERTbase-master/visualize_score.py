import pickle

import matplotlib.pyplot as plt

def plot_graph(file_name):
    with open('./scores/' + file_name,'rb') as f:
        train_acc = pickle.load(f)
        train_loss = pickle.load(f)
        valid_acc = pickle.load(f)
        valid_loss = pickle.load(f)

        valid_loss = [float(l.cpu()) for l in valid_loss]

    plt.subplot(2, 1, 1)
    plt.title('Accuracy')
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.legend(loc=5)

    plt.subplot(2, 1, 2)
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend(loc=5)

    plt.show()

if __name__ == '__main__':
    file_name = 'accloss9857.p'
    plot_graph(file_name)