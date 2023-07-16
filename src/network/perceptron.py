from activation import Activation
import numpy as np

import sys


class Perceptron:
    def __init__(self, fun: dict = None, base_fun: str = 'sigmoid') -> None:
        """
        Init for a multilayer perceptron

        Where the values of weights are calculated by ``2 * np.random.random([O_nodes, IN_nodes]) - 1``
        """

        # adding a list of wanted functions
        self.Functions = fun
        self.base_function = base_fun

        self.build = False

    def getWeights(self):
        return self.Synaptic_Weights

    def getBias(self):
        return self.Bias

    def getActivation(self):
        return self.Functions

    def getLearningRate(self):
        return self.Learning_Rate

    def getBias(self):
        return self.Bias

    def getLayers(self):
        return self.Layers

    def getLearningRate(self):
        return self.Learning_Rate

    def setLayersLoad(self, lay):
        self.Layers = lay
        return self

    def setLayers(self, input_nodes: int, hidden_layers: list, output_nodes: int):
        """
        Parameters
        ----------
        :param int input_nodes:
            Number of input `nodes`
        :param list of int hidden_layers:
            A list of numbers of neurons per each `hidden layer`
        :param int output_nodes:
            Number of output `nodes`

        Examples
        --------
        >>> p = Perceptron(...)
        >>> p.setLayers(2, [3,3], 2)
        >>> print(p.Synaptic_Weights[0])
        [[ 0.57519168  0.29013704]
        [-0.0346663  -0.26533394]
        [-0.43970131 -0.41541983]]
        >>> print(p.Synaptic_Weights[1])
        [[ 0.83238601  0.27162139 -0.06089101]
        [ 0.96001736  0.90498205 -0.1313054 ]
        [-0.48583136 -0.68419176 -0.86352844]]
        >>> print(p.Synaptic_Weights[2])
        [[ 0.51967537 -0.59031319 -0.2042105 ]]
        """

        self.Input_Nodes = input_nodes
        self.Output_Nodes = output_nodes
        self.Hidden_Layers = hidden_layers

        # Putting all layers into one list
        self.Layers = [input_nodes]

        for i in hidden_layers:
            self.Layers.append(i)

        self.Layers.append(output_nodes)

        return self

    def setWeights(self, weights = []):
        """"""

        self.Synaptic_Weights = weights

        # if no weights are given, then the weights are randomly generated
        if weights == []:
            for i in range(len(self.Layers)-1):
                # adding synaptic weights (limit from -1 to 1)
                self.Synaptic_Weights.append(
                    2 * np.random.random([self.Layers[i+1], self.Layers[i]]) - 1)

        return self

    def setBias(self, bias = []):
        """"""

        self.Bias = bias

        # if no bias is given, then the bias is randomly generated
        if bias == []:
            for i in range(len(self.Layers)-1):
                # adding bias (limit from -1 to 1)
                self.Bias.append(2 * np.random.random([self.Layers[i+1], 1]) - 1)

        return self

    def setLearningRate(self, lr: float):
        """"""

        self.Learning_Rate = lr

        return self

    def build_me(self):
        try:
            self.f_list = [self.base_function for i in self.Synaptic_Weights]
        except:
            print('ERROR: No weights set')
            sys.exit()


        # adds wanted functions to the list for later use
        if self.Functions:
            for key, elem in self.Functions.items():
                self.f_list[int(key)] = elem

        return self

    def FeedForwardFlex(self, inputs: np.ndarray, vb: int = 0):
        Outputs = []

        for i in range(len(self.Synaptic_Weights)):
            if vb:
                print(f'INPUTS PRE: ', inputs.shape)

            agg = np.dot(self.Synaptic_Weights[i], inputs) #+ self.Bias[i]

            if max(agg) != 0:
                np.divide(agg, np.max(agg))

            inputs = Activation(prefix=self.f_list[i]).function(agg)

            if vb:
                print(f'INPUTS POST: ', inputs.shape)

            Outputs.append(inputs)

        return Outputs

    def BackPropagationFlex(self, guesses: list, outputs: np.ndarray, ins: np.ndarray, no_layers: int = 0):

        if no_layers <= 0:
            stop_index = -1
        else:
            stop_index = len(guesses) - no_layers - 1

        errors = []

        errors.append(outputs - guesses[-1])

        for i in range(len(guesses)-1, stop_index, -1):
            # calculating gradient
            gradient = self.Learning_Rate * errors[-1] * Activation(prefix=self.f_list[i]).derivative(guesses[i])

            # clipping gradient between -1 and 1
            #grad = np.clip(grad, -1, 1)

            # adding to bias
            self.Bias[i] += gradient

            # adding to synaptic weights
            if i > 0:
                self.Synaptic_Weights[i] += np.dot(gradient, guesses[i-1].T)
            else:
                # if at last layer use Inputs for gradient
                self.Synaptic_Weights[i] += np.dot(gradient, ins.T)

            # adding new error (always using the last error) (this doesn't have to be a list)
            errors.append(np.dot(self.Synaptic_Weights[i].T, errors[-1]))

    def Train(self, inputs: np.ndarray, outputs: np.ndarray, iter: int, no_layers: int):
        """
        Method for training the network for `iter` amount of cycles

        This method is used by the user to train its network. Matching
        the number of parameters for each layer is important.

        Parameters
        ----------
        :param np.ndarray inputs:
            Array of input data
        :param np.ndarray outputs:
            Array of outputs to be tested
        :param int iter:
            Number of itterations

        Examples
        --------
        >>> training_inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
        >>> training_ouputs = np.array([[0], [0], [1], [1]])
        >>> p.Train(training_inputs, training_ouputs, 20000)
        """
        for i in range(iter):
            if i % 500 == 0:
                print(f"Training iteration: {i}")

            index = np.random.randint(0, len(inputs))

            # makes parameter data transposed
            t_inputs = np.array([inputs[index]]).T
            t_outputs = np.array([outputs[index]]).T

            guesses = self.FeedForwardFlex(t_inputs)
            self.BackPropagationFlex(guesses, t_outputs, t_inputs, no_layers)


if __name__ == "__main__":
    activation_dict = {
        '0': 'htan',
        '1': 'htan',
        '2': 'htan'
    }

    p = Perceptron(2, [3, 4], 3)

    training_inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    training_ouputs = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]])
    print('TI: ', training_inputs.shape)
    print('SW: ', p.Synaptic_Weights[2].shape)
    v = p.FeedForwardFlex(np.array([training_inputs[3]]).T, vb=1)

    # print(type(training_inputs))
    #p.Train(training_inputs, training_ouputs, 20000)

    #o = p.FeedForwardFlex(np.array([training_inputs[3]]).T, vb=1)
    # print(o[-1])
