import numpy as np
import os


def load_CIFAR_10(labeledExamples=None):
    U_train = None
    X_train = np.load(os.path.join('data', 'cifar_10_X_train.npy'), allow_pickle=True)
    Y_train = np.load(os.path.join('data', 'cifar_10_Y_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join('data', 'cifar_10_X_test.npy'), allow_pickle=True)
    Y_test = np.load(os.path.join('data', 'cifar_10_Y_test.npy'), allow_pickle=True)
    if labeledExamples is not None:
        U_train = X_train[labeledExamples:]
        X_train = X_train[:labeledExamples]
        Y_train = Y_train[:labeledExamples]
        return (X_train, Y_train), U_train, (X_test, Y_test)
    else:
        return (X_train, Y_train), (X_test, Y_test)

def split_batch(dataset, labeledExamples):
    labeled_dataset = [dataset[0][:, :labeledExamples], dataset[1][:, :labeledExamples]]
    unlabeled_dataset = [dataset[0][:, labeledExamples:], np.zeros(shape=dataset[1][:, labeledExamples:].shape)]
    return labeled_dataset, unlabeled_dataset

