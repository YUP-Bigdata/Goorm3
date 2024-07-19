import numpy as np


###################################################################
'''
returns accuracy
< compute_acc >
input: predictions, target_labels
output: accuracy
'''
###################################################################


def compute_acc(predictions, target_labels):
    return (np.array(predictions) == np.array(target_labels)).mean()