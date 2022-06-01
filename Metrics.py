from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
import numpy as np


def accuracy(true_labels, predict_labels):
    return accuracy_score(true_labels, predict_labels)


def ARI(true_labels, predict_labels):
    return adjusted_rand_score(true_labels, predict_labels)


def NMI(true_labels, predict_labels):
    return normalized_mutual_info_score(true_labels, predict_labels)


def F1(true_labels, predict_labels):
    return f1_score(true_labels, predict_labels, average='weighted')


# 关联矩阵
def RelevanceMatrix(labels):

    n_samples = len(labels)
    rm = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                rm[i, j] = 1
                rm[j, i] = 1
    return rm