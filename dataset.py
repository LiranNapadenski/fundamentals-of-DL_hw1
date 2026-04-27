import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import torch

def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def generete_data(PATH = 'cifar-10-batches-py', num_train = 5000, num_test = 1000):

    train_data = np.empty((0, 3072), dtype=np.uint8)
    train_labels = np.empty((0,), dtype=np.uint8)

    for i in range(1, 6):
        batch = unpickle(f"{PATH}/data_batch_{i}")
        train_data = np.vstack((train_data, batch[b'data']))
        train_labels = np.hstack((train_labels, batch[b'labels']))

    print("Data shape:", train_data.shape)
    print("Labels shape:", train_labels.shape)

    test_data = unpickle(f"{PATH}/test_batch")
    test_labels = np.array(test_data[b'labels'], dtype=np.uint8)
    test_data = np.array(test_data[b'data'], dtype=np.uint8)

    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)

    # Normalize data to be between 0 and 1
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    indices_train = np.random.choice(train_data.shape[0], size=num_train, replace=False)
    indices_test = np.random.choice(test_data.shape[0], size=num_test, replace=False)

    return train_data[indices_train], train_labels[indices_train], test_data[indices_test], test_labels[indices_test]