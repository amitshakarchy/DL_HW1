import tensorflow as tf
import numpy as np

import model_trainer
from time import time


def expand_y(y_data, num_classes=10):
    """
    Transform labels into one-hot encoded matrix
    :param y_data: the labels of the data
    :param num_classes: the number of unique classifications
    :return: one-hot encoded matrix
    """
    flatten_y = y_data.flatten().astype(int)
    expanded_y = np.zeros((flatten_y.shape[0], num_classes))
    expanded_y[np.arange(flatten_y.shape[0]), flatten_y] = 1
    return expanded_y


def reshape_data(data):
    """
    flatten 2-D samples to 1-D samples
    :param data: matrix (m, i, j), where m is the number of samples, and the 2-D samples are in shape (i, j)
    :return: matrix (m, i*j)
    """
    data_x, data_y = data
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2])
    data_y = np.expand_dims(data_y.T, axis=1)
    return data_x, data_y


def normalize(data):
    """
    Normalize the data which is in range of 0-255
    :param data:
    :return:  normalized data
    """
    data = data / 255
    return data


def pre_process():
    """
    Load the mnist dataset and prepare it to the model
    :return: dictionary contains the dataset
    """
    train_data, test_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_x, train_y = reshape_data(train_data)
    test_x, test_y = reshape_data(test_data)
    train_x, test_x = normalize(train_x), normalize(test_x)
    train_y, test_y= expand_y(train_y), expand_y(test_y)

    return {'train_x': train_x.T, 'train_y': train_y.T, 'test_x': test_x.T, 'test_y': test_y.T}


def run_config():
    """
    Set the model's configurations and run the model
    :return:
    """
    data_set = pre_process()
    layers_dim = [784, 20, 7, 5, 10]
    lr = 0.009
    iter_to_cost = 100
    batch_size = 256
    start = time()
    params, costs = model_trainer.L_layer_model(X=data_set['train_x'], Y=data_set['train_y'], layers_dims=layers_dim,
                                                learning_rate=lr, num_iterations=iter_to_cost, batch_size=batch_size)
    end = time()
    print(f"Test Accuracy: "
          f"{model_trainer.predict(X=data_set['test_x'], Y=data_set['test_y'], parameters=params) * 100:.2f}%")
    print(f'\nRunning Time: {end - start:.2f}s')


if __name__ == '__main__':
    np.random.seed(42)
    run_config()
