import pickle
import os
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

def _dataset(files):
    if type(files) == str:
        files = [files]
    X, y = [], []
    for f in files:
        with open(f, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            y.append(np.array(dict[b'labels']))
            X.append(np.array(dict[b'data'], dtype=np.float32).T)
    X = np.hstack(X) / 255
    y = np.array(y)
    y = np.hstack(y)
    Y = np.zeros((len(np.unique(y)), len(y)))
    Y[y, np.arange(len(y))] = 1
    idx = np.random.permutation(X.shape[1])
    X, Y = X[:, idx], Y[:, idx]
    return X, Y.T

def _create_CIFAR_10(path):
    train_files = [os.path.join(path, 'data_batch_{}'.format(batch)) for batch in range(1, 6)]
    test_file = os.path.join(path, 'test_batch')
    X_train, Y_train = _dataset(train_files)
    X_test, Y_test = _dataset(test_file)
    X_train = X_train.T.reshape((-1, 32, 32, 3), order='F')
    X_test = X_test.T.reshape((-1, 32, 32, 3), order='F')

    cf_mean, cf_std = np.array([0.4914, 0.4822, 0.4465]), np.array([0.2471, 0.2435, 0.2616])
    X_train = (X_train - cf_mean)/cf_std
    X_test = (X_test - cf_mean)/cf_std
    
    np.save(os.path.join(path, "cifar_10_X_train"), X_train)
    np.save(os.path.join(path, "cifar_10_Y_train"), Y_train)
    np.save(os.path.join(path, "cifar_10_X_test"), X_test)
    np.save(os.path.join(path, "cifar_10_Y_test"), Y_test)
    

if __name__ == "__main__":
    path_cifar10='./datasets/cifar-10-batches-py'
    _create_CIFAR_10('data')
    
