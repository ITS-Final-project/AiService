from network.config import *

from network.perceptron import Perceptron
from network.cnn import CNN

import numpy as np

from configuration.nnConfiguration import NNConfiguration

LABELS_MAP = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'P': 23,
    'R': 24,
    'T': 25,
    'U': 26,
    'W': 27,
    'X': 28
}

class ImageClassifier:

    def __init__(self):

        # config = load_config_from_file('./networks/NNTestLSigmoid4L7_500_500-8-93.67.npy')
        self.configFilePath = NNConfiguration.load_config()

        config = load_config_from_file(self.configFilePath)

        print("Loaded config: ", self.configFilePath)

        self.perceptron = config['perceptron']
        self.cnn = config['cnn']

    def classify(self, image):
        

        # Get image
        class_input = self.cnn.prepare_data([image], 1)

        print(class_input.shape)
        # Classify
        result = self.perceptron.FeedForwardFlex(class_input.T, vb=0)

        return list(LABELS_MAP.keys()) [list(LABELS_MAP.values()).index(np.argmax(result[-1]))]

    def train(self, image, label):


        # Get image
        class_input = self.cnn.prepare_data([image], 1)

        # Prepare label
        class_label = np.array([1 if i == label else 0 for i in range(10)])

        # Train
        self.perceptron.Train(class_input, class_label, 100, no_layers=2)

        result = self.perceptron.FeedForwardFlex(class_input.T, vb=0)

        print(result[-1])
        
        return list(LABELS_MAP.keys()) [list(LABELS_MAP.values()).index(np.argmax(result[-1]))]

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