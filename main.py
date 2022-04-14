from tensorflow.keras.datasets.mnist import load_data
import build_network


"""
4.	Use the code you wrote to classify the MNIST dataset and present a summary report
    a.	Load the dataset using the Keras code. Note that there is a predefined division between the train and test set.
        Use 20% of the training set as a validation set (samples need to be randomly chosen).
    b.	Run your network using the following configuration:
        •	4 layers (aside from the input layer), with the following sizes: 20,7,5,10
        •	Do not activate the batchnorm option at this point
        •	The input at each iteration needs to be “flattened” to a matrix of [m,784], where m is the number of samples
        •	Use a learning rate of 0.009
        •	Train the network until there is no improvement on the validation set (or the improvement is very small)
            for 100 training steps (this is the stopping criterion). Please include in the report the number of iterations
            and epochs needed to train your network. Also, specify the batch size.
    c.	Please include the following details in your report:
        •	The final accuracy values for the train, validation and test sets.
        •	The cost value for each 100 training steps. Please make sure that the index of the training step will
            also be included in the report. Print the values from the L_layer_model
    d.	All the information requested above will be included in a .docx file uploaded with the code.

"""
if '__name__' == 'main':
    x_train, y_train, x_test, y_test = load_data(path='mnist.npz')
    build_network.L_layer_model(x_train, y_train, layers_dims=[20, 7, 5, 10], learning_rate=0.009, num_iterations=100, batch_size=64)