from network.config import *

from network.perceptron import Perceptron
from network.cnn import CNN

import numpy as np

from configuration.nnConfiguration import NNConfiguration

class ImageClassifier:

    def __init__(self):

        # config = load_config_from_file('./networks/NNTestLSigmoid4L7_500_500-8-93.67.npy')
        self.configFilePath = NNConfiguration.load_config()

        config = load_config_from_file(self.configFilePath)

        print("Loaded config: ", self.configFilePath)

        self.perceptron = config['perceptron']
        self.cnn = config['cnn']

    def classify(self, image):


        print(image.shape)

        # Get image
        class_input = self.cnn.prepare_data([image], 1)

        print(class_input.shape)
        # Classify
        result = self.perceptron.FeedForwardFlex(class_input.T, vb=0)

        return np.argmax(result[-1])

    def train(self, image, label):


        # Get image
        class_input = self.cnn.prepare_data([image], 1)

        # Prepare label
        class_label = np.array([1 if i == label else 0 for i in range(10)])

        # Train
        self.perceptron.Train(class_input, class_label, 100, no_layers=2)

        result = self.perceptron.FeedForwardFlex(class_input.T, vb=0)

        print(result[-1])

        return np.argmax(result[-1])

    def getStructure(self):
        layers = self.perceptron.getLayers()
        learning_rate = self.perceptron.getLearningRate()
        activations = self.perceptron.getActivation()
        accuracy = self.configFilePath.split('-')[-1].split('.')[0]
        structure = {
            'layers': layers,
            'learning_rate': learning_rate,
            'activations': activations,
            'accuracy': accuracy
        }

        return structure