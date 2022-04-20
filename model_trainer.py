import backward
import forward
import numpy as np

BATCHNORM_USAGE = True


def generate_validation_data(data_x, data_y, validation_factor=0.2):
    """
    splitting the data to train and validation data by the validation_factor
    :param data_x: the x data to train
    :param data_y: the y data to train
    :param validation_factor: the percentage of the train data to be part of the validation data
    :return: a tuple of train_x, train_y, validation_x, validation_y
    """
    y_size = data_y.shape[0]
    combined_data = np.concatenate([data_x.T, data_y.T], axis=1)
    np.random.shuffle(combined_data)
    num_rows = int(validation_factor * combined_data.shape[0])
    validation, train = combined_data[:num_rows, :], combined_data[num_rows:, :]
    train_x, train_y = train[:, :-y_size], train[:, -y_size:]
    validation_x, validation_y = validation[:, :-y_size], validation[:, -y_size:]
    return train_x.T, train_y.T, validation_x.T, validation_y.T


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network.
    All layers but the last  have the ReLU activation function,
    and the final layer apply the softmax activation function
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch
    :return parameters: the parameters learnt by the system during the training
                        (the same parameters that were updated in the update_parameters function).
    :return costs: the values of the cost function.
                    One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).

    """
    stop_rate = 0.001

    train_x, train_y, validation_x, validation_y = generate_validation_data(X, Y)
    del X, Y

    combined_data = np.concatenate([train_x, train_y], axis=0)
    m = train_x.shape[1]

    num_batches = m // batch_size + (1 if m % batch_size else 0)

    params = forward.initialize_parameters(layers_dims)
    costs = []

    steps_cnt = 0
    while len(costs) < 2 or np.abs(costs[-2] - costs[-1]) > stop_rate:
        if not steps_cnt % num_batches:
            np.random.shuffle(combined_data.T)
            batches = np.array_split(combined_data, indices_or_sections=num_batches, axis=1)

        X_batch = batches[steps_cnt % num_batches][0:train_x.shape[0], :]
        Y_batch = batches[steps_cnt % num_batches][train_x.shape[0]:, :]

        prediction, caches = forward.L_model_forward(X_batch, params,
                                                     use_batchnorm=BATCHNORM_USAGE)
        grads = backward.L_model_backward(prediction, Y_batch, caches)
        params = backward.update_parameters(params, grads, learning_rate)

        steps_cnt += 1

        if not steps_cnt % num_iterations:
            prediction, _ = forward.L_model_forward(validation_x, params,
                                                    use_batchnorm=BATCHNORM_USAGE)
            cost = forward.compute_cost(prediction, validation_y)
            costs.append(cost)
            print(f'\tStep number: {steps_cnt} - Cost {cost:.3f}')

    steps_str = '' if not steps_cnt % num_batches else f' and {steps_cnt % num_batches} steps'
    print(f'\nRan over {steps_cnt // num_batches} epochs' + steps_str)

    print(f"\nTrain Accuracy: {predict(X=train_x, Y=train_y, parameters=params)*100:.2f}%")
    print(f"Validation Accuracy: {predict(X=validation_x, Y=validation_y, parameters=params) * 100:.2f}%")

    return params, costs


def predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and
    calculates the accuracy of the trained neural network on the data.

    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return accuracy: the accuracy measure of the neural net on the provided data
                    (i.e. the percentage of the samples for which the correct
                     label receives the hugest confidence score).
                     Using the softmax function to normalize the output values
    """
    m = X.shape[1]
    prediction, caches = forward.L_model_forward(X, parameters,
                                                 use_batchnorm=BATCHNORM_USAGE)
    prediction_arg_max = np.argmax(prediction, axis=0)
    label_arg_max = np.argmax(Y, axis=0)
    correct_predictions = np.sum(prediction_arg_max == label_arg_max)
    accuracy = correct_predictions / m
    return accuracy
