import numpy as np
import random


class Layer:
    def __init__(self, number_of_neurons, number_of_input):
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.uniform(-1, 1, (number_of_input, number_of_neurons))
        self.bias = np.random.uniform(-1, 1, (number_of_neurons,))
        print(self.weights.shape)

        self.sum = lambda x: x.dot(self.weights) + self.bias
        self.output = lambda x: self.f(self.sum(x))
        self.f = lambda x: 1/(1+np.exp(-x))
        self.derivative_f = lambda x: np.exp(-x) / ((1 + np.exp(-x))**2)


def split_data(train_set, test_set, data_table):
    # TODO Split data so that it has 4:1 ratio for train:test
    for row in data_table:
        if random.random() > 0.8:
            test_set.append(row)
        else:
            train_set.append(row)


def load_data():
    with open("MNISTnumImages5000.txt", 'r') as in_file:
        input_data_set = []
        for row in in_file:
            new_row = []
            row = row.split('\t')
            for item in row:
                new_row.append(float(item))
            input_data_set.append(new_row)
        return np.asarray(input_data_set)


def input_neural_network_parameters():
    hidden_layers = int(input("How many hidden layers in the neural network? "))
    parameters = []
    for x in range(hidden_layers):
        neurons = int(input("How many neurons in Hidden Layer " + str(x+1) + "? "))
        parameters.append(neurons)
    return parameters


def load_labels():
    with open("MNISTnumLabels5000.txt", 'r') as in_file:
        answer_set = []
        for row in in_file:
            new_row = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            row = int(row.rstrip())
            new_row[0][row] = 1
            answer_set.append(new_row)
        return np.asarray(answer_set)


def update_weights(alpha, p_deltas, deltas, network):
    if p_deltas is None:
        print()
    else:
        for layers in network:
            layers.weights = layers.weights + deltas + alpha * p_deltas


def predict(network, test_set, labels, epoch, desired_outcome, actual_outcome):
    correct = 0
    for i in range(0, test_set.shape[0]):
        output_of_layers = [dataset[i]]
        for layers in network:
            output_of_layers.append(layers.output(output_of_layers[-1]))
            if epoch == 99 and layers is network[-1]:
                actual = np.argmax(output_of_layers[-1])
                actual_outcome.append(actual)
                desired = np.argmax(labels[i])
                desired_outcome.append(desired)
            if np.argmax(output_of_layers[-1]) == np.argmax(labels[i]) and layers is network[-1]:
                correct += 1
    hit_rate = correct / test_set.shape[0]
    print("Hit Rate: "+str(hit_rate) + "\t Epoch: " + str(epoch + 1))


if __name__ == "__main__":
    dataset = load_data()
    answers = load_labels()

    test_answers = answers[0:1000]
    test_set = dataset[0:1000]
    dataset = dataset[:-1000]
    answers = answers[:-1000]

    # neuron_count = input_neural_network_parameters()

    learning_rate = 0.15
    momentum_rate = 0.5
    np.random.seed(1)
    network = []
    hidden_neurons = 150

    network.append(Layer(hidden_neurons, dataset.shape[1]))
    network.append(Layer(10, hidden_neurons))
    # network.append(Layer(784, hidden_neurons))
    epochs = 100
    test_desired_outcome = []
    test_actual_outcome = []
    train_desired_outcome = []
    train_actual_outcome = []
    for e in range(epochs):
        correct = 0
        previous_deltas = None
        # One Epoch of training
        # For each row in the dataset, get the output and adjust the weights
        for i in range(0, dataset.shape[0]):
            row = dataset[i]
            output_of_layers = [dataset[i]]

            # Forward Propagate input
            for layer in network:
                output_of_layers.append(layer.output(output_of_layers[-1]))

            # Go through outputs backwards
            for outputs in range(len(output_of_layers) - 1, 1, -1):
                error = None
                momentum = 0

                # LMS
                deltas = []
                if outputs == len(output_of_layers) - 1:
                    layer_output = output_of_layers[outputs]
                    layer_input = output_of_layers[outputs - 1]
                    current_layer = network[outputs - 1]

                    error = answers[i] - layer_output
                    df = np.array([current_layer.derivative_f(layer_input)]).T
                    layer_delta = learning_rate * df * error
                    deltas.append(layer_delta)

                    # Update weights and biases
                    if previous_deltas is None:
                        momentum = 0
                    else:
                        momentum = momentum_rate * previous_deltas[0]
                    current_layer.weights = current_layer.weights + layer_input.reshape(-1, 1) * layer_delta + momentum
                    current_layer.bias = (current_layer.bias + layer_delta)[0]
                    if e == 99:
                        actual_result = np.argmax(layer_output)
                        desired_result = np.argmax(answers[i])
                        train_actual_outcome.append(actual_result)
                        train_desired_outcome.append(desired_result)
                    if np.argmax(layer_output) == np.argmax(answers[i]):
                        correct += 1

                else:
                    layer_output = output_of_layers[outputs]
                    layer_input = output_of_layers[outputs-1]

                    current_layer = network[outputs - 1]
                    layer_ahead = network[outputs]

                    error = layer_ahead.weights * deltas[0]
                    df = np.array([current_layer.derivative_f(layer_input)]).T
                    layer_delta = learning_rate * df * error
                    deltas.insert(0, layer_delta)
                    if previous_deltas is None:
                        momentum = 0
                    else:
                        momentum = momentum_rate * previous_deltas[0]
                    current_layer.weights = current_layer.weights + layer_input.reshape(-1, 1) * layer_delta + momentum
                    current_layer.bias = (current_layer.bias + layer_delta)[0]
            previous_deltas = deltas
        predict(network, test_set, test_answers, e, test_desired_outcome, test_actual_outcome)
        # print(correct/5000)
        # print(e)
    print("Test A")
    print(test_actual_outcome)
    print("Test D")
    print(test_desired_outcome)
    print("Train D")
    print(train_desired_outcome)
    print("Train A")
    print(train_actual_outcome)
    pass
