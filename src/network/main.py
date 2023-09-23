import sys
import array
from perceptron import Perceptron
from activation import Activation
from cnn import CNN, CNN_Layer
from config import load_config_from_file, save_config_to_file
import cv2

from mnist import MNIST as MNISTLoader

import numpy as np
import os

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

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

def load_letters(images_folder: str):
    images = []
    labels = []

    print("Loading letters...")
    for image_file in os.listdir(images_folder):
        letter = image_file[0]

        if letter == 'I' or letter == 'O':
            continue

        image = cv2.imread(images_folder + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        images.append(image)
        labels.append(letter)

    # Make array 
    labels = array.array('u', labels)

    data = {
        "training": {
            "images": images,
            "labels": labels
        },
        "testing": {
            "images": images,
            "labels": labels
        }
    }

    # print(data['training']['labels'])

    return data

def load_workdata():
    images = []

    # Load MNIST data (must be opened from src folder)
    mndata = MNISTLoader('./network/samples')
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    labels_testing = array.array("u", [str(c) for c in labels_testing])
    labels_training  = array.array("u", [str(c) for c in labels_training])

    data = {
        "training": {
            "images": images_training,
            "labels": labels_training
        },
        "testing": {
            "images": images_testing,
            "labels": labels_testing
        }
    }

    # print(data['training']['labels'])

    return data

def reshape_images(images, width, height):
    reshaped_images = []
    for image in images:
        reshaped_images.append(np.array(image).reshape(width, height))

    return reshaped_images

def get_random_images(images, labels, count):
    
    random_indexes = np.random.randint(0, len(images), count)

    images = np.array([images[i] for i in random_indexes])
    labels = np.array([labels[i] for i in random_indexes])

    return images, labels

def get_labels(numbered_labels):
    labels = []
    for label in numbered_labels:
        l_array = np.zeros(29)
        l_array[LABELS_MAP[str(label)]] = 1
        labels.append(l_array)
        # labels.append(np.array([1 if i == label else 0 for i in range(10)]))

    return labels
    

def get_accuracy(model, images, lables, test_count):
    hits = 0
    for i in range(test_count):
        results = model.FeedForwardFlex(np.array([images[i]]).T, vb=0)
        a = results[-1]
        a = np.where(a == np.max(a), 1, 0)
        val = a.flatten()

        if (val == lables[i]).all():
            hits += 1

    accuracy = round(hits/test_count*100, 2)

    return accuracy

def cleanup_networks(dir):
    # Remove 2 oldest networks
    networks = os.listdir(dir)
    networks.sort()

    if len(networks) > 4:
        os.remove(dir + networks[0])
        os.remove(dir + networks[1])


def main():
    
    # define filters
    l1_filter = np.zeros((4, 3, 3))

    l1_filter[0, :, :] = np.array([[
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]]])

    l1_filter[1, :, :] = np.array([[
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]]])

    l1_filter[2, :, :] = np.array([[
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]]])

    l1_filter[3, :, :] = np.array([[
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]]])


    # Define layers
    layer_1 = CNN_Layer(l1_filter, 'relu', 'SamePadding', 'max')
    layer_2 = CNN_Layer(l1_filter, 'relu', 'SamePadding', 'max')

    cnn_model = CNN([layer_1, layer_2])

    
    prepare = False
    if prepare:

        print("Preparing set to true, preparing data...")

        print("Loading data...")
        # load images and labels
        letters_data = load_letters('./letters/')
        data = load_workdata()

        # Adding letters data to overall data
        letters_data['training']['images'] += data['training']['images']
        letters_data['testing']['images'] += data['testing']['images']

        letters_data['training']['labels'] += data['training']['labels']
        letters_data['testing']['labels'] += data['testing']['labels']

        # Have to add letter to the beginning
        data['training']['images'] = letters_data['training']['images']
        data['testing']['images'] = letters_data['testing']['images']

        data['training']['labels'] = letters_data['training']['labels']
        data['testing']['labels'] = letters_data['testing']['labels']

        print("Reshaping images...")
        # reshape images (from 1d to 2d) => (784,) to (28, 28)
        data['training']['images'] = reshape_images(data['training']['images'], IMAGE_WIDTH, IMAGE_HEIGHT)
        data['testing']['images'] = reshape_images(data['testing']['images'], IMAGE_WIDTH, IMAGE_HEIGHT)

        print("Fixing labels...")
        # get labels => 1 to [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (one hot encoding)
        data['training']['labels'] = get_labels(data['training']['labels'])
        data['testing']['labels'] = get_labels(data['testing']['labels'])

        print("Feature maps...")
        # put images through CNN => (28, 28) to (28, 28, 4) => Feture maps
        training_input = cnn_model.prepare_data(data['training']['images'], 2000, 1)

        print("Testing feature maps ...")
        testing_input = cnn_model.prepare_data(data['testing']['images'], 2000, 1)


        print("Saving data...")
        # save data
        np.save("preparedTestingData", { "input": testing_input, "lables": data['testing']['labels'] } )
        np.save("preparedTrainingData", { "input": training_input, "lables": data['training']['labels'] } )


    print("Loading data (images)...")
    # load input values
    testing_input = np.load("preparedTestingData.npy", allow_pickle=True).item()['input']
    training_input = np.load("preparedTestingData.npy", allow_pickle=True).item()['input']

    print("Loading data (labels)...")
    # load lables
    testing_lables = np.load("preparedTestingData.npy", allow_pickle=True).item()['lables']
    training_lables = np.load("preparedTestingData.npy", allow_pickle=True).item()['lables']


    # Define perceptron
    activation_dict = {
        '0': 'sigmoid',
        '1': 'sigmoid',
        '2': 'sigmoid',
        '3': 'sigmoid',
        '4': 'sigmoid'
    }

    perc = Perceptron(fun=activation_dict, base_fun='lrelu')\
                                                .setLayers(
                                                    training_input.shape[1], [
                                                    training_input.shape[1]*2,
                                                    training_input.shape[1]//4,
                                                    training_input.shape[1]//8,
                                                    training_input.shape[1]//16],
                                                    29)\
                                                .setLearningRate(0.001)\
                                                .setBias()\
                                                .setWeights()\
                                                .build_me()
    
    # A B C D E F G H J K L M N P R T U W = 18 + 10 = 28

    print("Saving config...")
    # Save model to file
    

    eras = 1000
    steps = 1500

    max_accuracy = 0
    best_era = 0

    trained_layers = 0

    total = 1000
    train_image_count = 1500
    testing_image_count = total - train_image_count
    counter = 0

    print("Training images: \n")
    print("Eras: " + str(eras))
    print("Steps: " + str(steps))
    
    for e in range(eras):
        print("Current era: " + str(e + 1))
        print("Training layers: ", trained_layers)

        # training_input, training_labels = get_random_images(training_input, train_image_count)

        perc.Train(training_input, training_lables, steps, no_layers = trained_layers)

        accuracy = get_accuracy(perc, testing_input, testing_lables, 1000)

        if accuracy >= max_accuracy:
            counter += 1
            max_accuracy = accuracy
            best_era = e
            network_name = "NNTestLSigmoidLetters4L"+str(counter)+"_500_500-" + str(e) + "-" + str(round(accuracy, 2))
            # np.save("./networks/" + network_name, np.array(perc.getWeights(), dtype=object))

            save_config_to_file("./networks/" + network_name, perc, cnn_model, accuracy)
            cleanup_networks("./networks/")

        print("Era: " + str(e + 1) + " Accuracy: " + str(accuracy) + "%")
        print("Max accuracy: " + str(max_accuracy) + "% in era: " + str(best_era + 1) + "\n")


if __name__ == '__main__':
    main()