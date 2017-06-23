import random
import math
import numpy
import copy
import itertools

class Neuron:
    def __init__(self, parent, act_func, is_out, is_in, in_):
        self.network = parent
        self.activation_function_index = act_func
        self.inputs = in_
        self.is_output = is_out
        self.is_input = is_in
        self.weights = list()
        self.has_been_calculated = False
        self.output_value = 0
        self.error_has_been_calculated = False
        self.error = 0
        self.initialize_weights()
        self.dependents = []

    def set_dependents(self):
        for n in self.network.neurons:
            if self.network.neurons.index(self) in n.inputs:
                self.dependents.append(self.network.neurons.index(n))

    def initialize_weights(self):
        self.weights.append(self.network.bias_theta)
        for i in self.inputs:
            w = random.uniform(-self.network.starting_weight_boundary,
                               self.network.starting_weight_boundary)
            self.weights.append(w)

    def induced_local_field(self):
        if self.has_been_calculated:
            return 0
        sum_ = self.network.bias_input * self.weights[0]
        count = 1
        for i in self.inputs:
            weighted_input = (self.network.neurons[i].output() *
                              self.weights[count])
            sum_ += weighted_input
            count += 1
        return sum_

    def logistic_activation_function(self, induced_local_field_):
        if self.has_been_calculated: return self.output_value
        self.output_value = 1 / (1 + math.exp(-induced_local_field_))
        self.has_been_calculated = True
        return self.output_value

    def logistic_activation_function_derivative(self, induced_local_field_):
        activation_function = (self.logistic_activation_function
                               (induced_local_field_))
        return activation_function * (1 - activation_function)

    activation_functions = {
        0: logistic_activation_function
    }
    activation_functions_derivatives = {
        0: logistic_activation_function_derivative
    }

    def output(self):
        if self.is_input:
            return self.network.input_data[self.inputs[0]]
        else:
            return (self.activation_functions[self.activation_function_index]
                    (self, self.induced_local_field()))

    def unit_error(self, expected_values):
        if self.error_has_been_calculated:
            return self.error
        if self.is_output:
            self.error = ((expected_values[self.network.output_neurons_index.
                           index(self.network.neurons.index(self))] - self.
                           output()) * self.activation_functions_derivatives
                          [self.activation_function_index](self, self.
                                                           induced_local_field()))
        else:
            sum_ = 0
            for i in self.dependents:
                sum_ += (self.network.neurons[i].unit_error(expected_values) *
                         self.network.neurons[i].weights[self.network.neurons
                         [i].inputs.index(self.network.neurons.index(self))])
            self.error = (self.activation_functions_derivatives[self.
                          activation_function_index](self, self.
                                                     induced_local_field()) * sum_)
        self.error_has_been_calculated = True
        return self.error

class NeuralNetwork:
    def __init__(self, in_):
        self.bias_input = -1
        self.bias_theta = 0.5
        self.starting_weight_boundary = 1
        self.neurons = list()
        self.inputs = in_
        self.output_neurons_index = list()
        self.input_neurons_index = list()
        self.initialize_neurons()
        self.bprop = Backpropagation(self)
        self.input_data = []
        self.output_data = []

    def initialize_neurons(self):
        for i in inputs:
            # n = Neuron(self,i.activation_function,i.is_output,i.is_in,i.inputs)
            n = Neuron(self, 0, i[1], i[2], i[3:len(i)])
            self.neurons.append(n)
        for n in self.neurons:
            n.set_dependents()
            if n.is_output:
                self.output_neurons_index.append(self.neurons.index(n))
            if n.is_input:
                self.input_neurons_index.append(self.neurons.index(n))

    def feedforward(self):
        outputs = []
        for n in self.neurons:
            if n.is_output:
                outputs.append(n.output())
        return outputs

    def network_error(self):
        outputs = self.feedforward()
        sum_ = 0
        for idx in range(len(outputs) - 1):
            sum_ += pow(self.output_data[idx] - outputs[idx], 2)
        return sum_ / 2, outputs

class Backpropagation:
    def __init__(self, parent):
        self.network = parent
        self.initial_learning_rate = 0.7
        self.learning_rate_k = 1
        self.momentum_rate = 1
        self.learning_rate = 1

    def update_learning_rate(self, iteration):
        return self.learning_rate_k * math.exp(
            -self.initial_learning_rate * iteration)

    def update_weights(self):
        for n in reversed(self.network.neurons):
            n.weights[0] += (self.learning_rate * n.error * self.network.
                             bias_input)
            count = 1
            for i in n.inputs:
                n.weights[n.inputs.index(i) + 1] += (self.learning_rate *
                                                     n.error * self.network.
                                                     neurons[i].output())
                count += 1

    def run(self, iteration):
        #self.learning_rate = self.update_learning_rate(iteration)
        for n in self.network.neurons:
            if n.is_input:
                continue
            if n.error_has_been_calculated:
                continue
            n.unit_error(self.network.output_data)
        self.update_weights()


def k_fold(attributes, outputs, k, number_of_classes):
    classes = list()
    classes_counter = list()
    folded_attributes = list()
    folded_outputs = list()
    for i in range(0, k):
        folded_attributes.append(list())
        folded_outputs.append(list())
    for i in range(0, number_of_classes):
        classes_counter.append(0)
    for output in outputs:
        if output not in classes:
            classes.append(output)
        classes_counter[classes.index(output)] += 1

    for class_index in range(0, len(classes_counter)):
        fold = 0
        while classes_counter[class_index] > 0:
            index = random.randint(0, len(outputs) - 1)
            while outputs[index] != classes[class_index]:
                index = random.randint(0, len(outputs) - 1)
            folded_attributes[fold].append(attributes[index])
            folded_outputs[fold].append(outputs[index])
            del attributes[index]
            del outputs[index]
            fold += 1
            if fold >= k:
                fold = 0
            classes_counter[class_index] -= 1
    return folded_attributes, folded_outputs

def extract_testing_data(attributes, outputs, ratio, number_of_classes):
    classes = list()
    classes_counter = list()
    extracted_attributes = list()
    extracted_outputs = list()
    for i in range(0, number_of_classes):
        classes_counter.append(0)
    for output in outputs:
        if output not in classes:
            classes.append(output)
        classes_counter[classes.index(output)] += 1

    classes_counter = [int(ratio * c) for c in classes_counter]

    for class_index in range(0, len(classes_counter)):
        while classes_counter[class_index] > 0:
            index = random.randint(0, len(outputs) - 1)
            if outputs[index] == classes[class_index]:
                extracted_attributes.append(attributes[index])
                extracted_outputs.append(outputs[index])
                del attributes[index]
                del outputs[index]
                classes_counter[class_index] -= 1

    return attributes, outputs, extracted_attributes, extracted_outputs

def normalize(attributes, number_of_attributes):
    for i in range(number_of_attributes):
        min_value = min(a[i] for a in attributes)
        max_value = max(a[i] for a in attributes)
        for attribute in attributes:
            if min_value == max_value:
                attribute[i] = 0
            else:
                attribute[i] = (attribute[i] - min_value) / (max_value - min_value)
    return attributes

if __name__ == "__main__":
    # input neurons with respective inputs and if is output or not
    # pattern: neuron_index is_output is_input input_neurons_indexes
    data_attributes = list()
    data_outputs = list()
    training_data_attributes = list()
    training_data_outputs = list()
    validation_data_attributes = list()
    validation_data_outputs = list()
    test_data_attributes = list()
    test_data_outputs = list()
    inputs = []
    att = []

    # testing_ratio = 1
    testing_ratio = 0.2
    # while testing_ratio >= 1:
    #     print("Input the desired testing ratio:")
    #     testing_ratio = float(input())

    # k_folds = 0
    k_folds = 10
    # while k_folds < 2:
    #     print("How many folds for k-fold cross-validation? ")
    #     k_folds = int(input())

    # training_folds = 0
    training_folds = 8
    # while training_folds < 1 or training_folds >= k_folds:
    #     print("How many folds for training? ")
    #     training_folds = int(input())

    # print("What data set do you wish to use? ")
    # data_set_name = input()
    data_set_name = "segmentation"

    for line in open("{data_set_name}/{data_set_name}_topology".format(
            data_set_name=data_set_name), 'r').readlines():
        values = line.split()
        row = [int(value) for value in values]
        inputs.append(row)
    print(inputs)

    for line in open(
            "{data_set_name}/{data_set_name}_attributes".format(
                data_set_name=data_set_name), 'r').readlines():
        row = line
        att.append(row)
    print(att)
    number_of_attributes = int(att[0])
    number_of_outputs = int(att[1])
    print(number_of_attributes)
    print(number_of_outputs)

    for line in open("{data_set_name}/{data_set_name}_data".format(
            data_set_name=data_set_name), 'r').readlines():
        row = line
        values = row.split()
        row = [float(value) for value in values]
        att_row = row[:number_of_attributes]
        output_row = row[number_of_attributes:]
        data_attributes.append(att_row)
        data_outputs.append(output_row)

    data_attributes = normalize(data_attributes, number_of_attributes)

    data_attributes, data_outputs, test_data_attributes, test_data_outputs = \
        extract_testing_data(data_attributes, data_outputs, testing_ratio,
                             number_of_outputs)

    # for line in open("{data_set_name}/{data_set_name}_training.txt".format(
    #         data_set_name=data_set_name), 'r').readlines():
    #     row = line
    #     values = row.split()
    #     row = [float(value) for value in values]
    #     att_row = row[:number_of_attributes]
    #     output_row = row[number_of_attributes:]
    #     training_data_attributes.append(att_row)
    #     training_data_outputs.append(output_row)
    # print(training_data_attributes)
    # print(training_data_outputs)

    # for line in open(
    #         "{data_set_name}/{data_set_name}_validation.txt".format(
    #             data_set_name=data_set_name), 'r').readlines():
    #     row = line
    #     values = row.split()
    #     row = [float(value) for value in values]
    #     att_row = row[:number_of_attributes]
    #     output_row = row[number_of_attributes:]
    #     validation_data_attributes.append(att_row)
    #     validation_data_outputs.append(output_row)
    # print(validation_data_attributes)
    # print(validation_data_outputs)

    # for line in open("{data_set_name}/{data_set_name}_test.txt".format(
    #         data_set_name=data_set_name), 'r').readlines():
    #     row = line
    #     values = row.split()
    #     row = [float(value) for value in values]
    #     att_row = row[:number_of_attributes]
    #     output_row = row[number_of_attributes:]
    #     test_data_attributes.append(att_row)
    #     test_data_outputs.append(output_row)
    # print(test_data_attributes)
    # print(test_data_outputs)

    total_training_errors = list()
    total_validation_errors = list()
    total_testing_errors = list()

    error_threshold = 0.001
    max_iterations = 2000
    min_iteration = max_iterations / 50
    validation_error_range = int(max_iterations / 100)
    number_of_tests = 10

    # training & validation
    # error = float('Inf')
    validation_error = 0
    for test in range(0, number_of_tests):

        # construct network
        nn = NeuralNetwork(inputs)

        best_iteration = 0
        best_iteration_error = float('Inf')
        best_network = copy.deepcopy(nn)
        test_training_errors = list()
        test_validation_errors = list()
        folded_attributes, folded_outputs = k_fold(data_attributes,
                                                   data_outputs, k_folds,
                                                   number_of_outputs)

        for iteration in range(0, max_iterations):

            # select folds for training and validation
            used_fold_indices = list()
            for i in range(0, training_folds):
                index = random.randint(0, k_folds - 1)
                while index in used_fold_indices:
                    index = random.randint(0, k_folds - 1)
                    used_fold_indices.append(index)
                training_data_attributes += folded_attributes[index]
                training_data_outputs += folded_outputs[index]
            for i in range(0, k_folds - training_folds):
                index = random.randint(0, k_folds - 1)
                while index in used_fold_indices:
                    index = random.randint(0, k_folds - 1)
                    used_fold_indices.append(index)
                validation_data_attributes += folded_attributes[index]
                validation_data_outputs += folded_outputs[index]
            print("..")
            training_error = 0
            for index in range(len(training_data_attributes)):
                for neuron in nn.neurons:
                    neuron.has_been_calculated = False
                    neuron.error_has_been_calculated = False
                nn.input_data = training_data_attributes[index]
                nn.output_data = training_data_outputs[index]
                training_error += nn.network_error()[0]
                nn.bprop.run(iteration)
            training_error /= len(training_data_attributes)
            test_training_errors.append(training_error)
            validation_error = 0
            for index in range(len(validation_data_attributes)):
                for neuron in nn.neurons:
                    neuron.has_been_calculated = False
                    neuron.error_has_been_calculated = False
                nn.input_data = validation_data_attributes[index]
                nn.output_data = validation_data_outputs[index]
                validation_error += nn.network_error()[0]
            validation_error /= len(validation_data_attributes)
            test_validation_errors.append(validation_error)
            if validation_error < best_iteration_error:
                best_iteration = iteration
                best_iteration_error = validation_error
                best_network = copy.deepcopy(nn)
            elif iteration > min_iteration:
                if not validation_error < min(test_validation_errors[
                                              -validation_error_range:-1]):
                    print(best_iteration)
                    break

        nn = copy.deepcopy(best_network)

        test_testing_error = 0
        # testing
        for index in range(len(test_data_attributes)):
            for neuron in nn.neurons:
                neuron.has_been_calculated = False
                neuron.error_has_been_calculated = False
            nn.input_data = test_data_attributes[index]
            nn.output_data = test_data_outputs[index]
            print("Expected: ", nn.output_data)
            testing_error, obtained = nn.network_error()
            test_testing_error += testing_error
            print("Obtained: ", obtained)
        test_testing_error /= len(test_data_attributes)

        total_training_errors.append(test_training_errors)
        total_validation_errors.append(test_validation_errors)
        total_testing_errors.append(test_testing_error)

    output_file = open("{data_set_name}/{data_set_name}_results.txt".format(
        data_set_name=data_set_name), 'w')
    for test in range(0, number_of_tests):
        for iteration in total_training_errors[test]:
            output_file.write("{error} ".format(error=iteration))
        output_file.write("\n")
        for iteration in total_validation_errors[test]:
            output_file.write("{error} ".format(error=iteration))
        output_file.write("\n")
        output_file.write("{error}".format(error=total_testing_errors[test]))
        output_file.write("\n\n")
    total_training_errors_list = list(
        itertools.chain.from_iterable(total_training_errors))
    total_validation_errors_list = list(
        itertools.chain.from_iterable(total_validation_errors))
    total_testing_errors_list = total_testing_errors
    output_file.write("Minimum training error: {error}\n".format(
        error=min(total_training_errors_list)))
    output_file.write("Minimum validation error: {error}\n".format(
        error=min(total_validation_errors_list)))
    output_file.write("Minimum testing error: {error}\n".format(
        error=min(total_testing_errors_list)))
    output_file.write("Maximum training error: {error}\n".format(
        error=max(total_training_errors_list)))
    output_file.write("Maximum validation error: {error}\n".format(
        error=max(total_validation_errors_list)))
    output_file.write("Maximum testing error: {error}\n".format(
        error=max(total_testing_errors_list)))
    output_file.write("Mean training error: {error}\n".format(
        error=numpy.mean(total_training_errors_list)))
    output_file.write("Mean validation error: {error}\n".format(
        error=numpy.mean(total_validation_errors_list)))
    output_file.write("Mean testing error: {error}\n".format(
        error=numpy.mean(total_testing_errors_list)))
    output_file.write("Median training error: {error}\n".format(
        error=numpy.median(total_training_errors_list)))
    output_file.write("Median validation error: {error}\n".format(
        error=numpy.median(total_validation_errors_list)))
    output_file.write("Median testing error: {error}\n".format(
        error=numpy.median(total_testing_errors_list)))
    output_file.write("Stddev training error: {error}\n".format(
        error=numpy.std(total_training_errors_list)))
    output_file.write("Stddev validation error: {error}\n".format(
        error=numpy.std(total_validation_errors_list)))
    output_file.write("Stddev testing error: {error}\n".format(
        error=numpy.std(total_testing_errors_list)))
    output_file.close()
