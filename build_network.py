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
    pass


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
