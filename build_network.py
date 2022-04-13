import forward_propagation as forward
import backward_propagation as backward
import numpy as np
from numpy import random

"""PARAMS"""
USE_BATCHNORM = False  # used in L_layer_model function
VALIDATION_FRAC = 0.2
STOPPING_CRITERIA = 100  # 100 training steps with no change


def train_validation_split(X, Y):
    all_data = np.concatenate([X, Y], axis=0)  # concat rows before shuffling
    random.shuffle(all_data)
    train, validation = all_data[VALIDATION_FRAC * len(all_data):, :], all_data[:VALIDATION_FRAC * len(all_data), :]
    return train, validation


def split_x_y(data, y_size):
    x, y = data[:, :y_size], data[:, y_size:]
    return x, y


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function, and the final
    layer will apply the softmax activation function. The size of the output layer should be equal to the number of
    labels in the data. Please select a batch size that enables your code to run well (i.e. no memory overflows while
    still running relatively fast).
    Hint: the function should use the earlier functions in the following order: initialize -> L_model_forward ->
    compute_cost -> L_model_backward -> update parameters

    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate:
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch.
    :return:
        parameters – the parameters learnt by the system during the training (the same parameters that were updated in
        the update_parameters function).
        costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved
        after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """
    costs = list()
    parameters = forward.initialize_parameters(layers_dims)
    y_cols_n = Y.shape[0]
    training_steps_counter = 0

    train, validation = train_validation_split(X, Y)
    x_val, y_val = split_x_y(validation, y_cols_n)
    x_train, y_train = split_x_y(train, y_cols_n)
    examples_num = train.shape[1]
    n_batches = examples_num // batch_size

    while (training_steps_counter < STOPPING_CRITERIA) and (training_steps_counter < num_iterations):
        random.shuffle(train)
        batches = np.split(train, n_batches, axis=1)
        for batch in batches:
            x_b, y_b = split_x_y(batch, y_cols_n)
            AL, caches = forward.L_model_forward(x_b, parameters, USE_BATCHNORM)
            grads = backward.L_model_backward(AL, Y, caches)
            parameters = backward.Update_parameters(parameters, grads, learning_rate)
            training_steps_counter += 1

        if training_steps_counter % 100 == 0:
            A_val, _ = forward.L_model_forward(x_val, parameters, USE_BATCHNORM)
            cost = forward.compute_cost(A_val, y_val)
            costs.append(cost)
            print(
                f"Validation: step #{training_steps_counter}/{num_iterations}, acc: {Predict(x_val, y_val, parameters)}")
        print(
            f"Training: step #{training_steps_counter}/{num_iterations}: acc: {Predict(x_train, y_train, parameters)}")

    return parameters, costs


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return:
        accuracy – the accuracy measure of the neural net on the provided data
        (i.e. the percentage of the samples for which the correct label receives the hughest confidence score).
        Use the softmax function to normalize the output values.
    """
