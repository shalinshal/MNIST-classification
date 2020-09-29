# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:39:23 2019

@author: Shalin
"""
import sys,os
import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
import struct
from array import array
from cnn import CNN
import pandas as pd

sys.path.append('../')


DATA_PATH = '../dataset/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',\
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')
#    parser.add_argument('--load', action='store_true',\
#                        help='Load dataset')
    parser.add_argument('--activation',nargs='*', action='store', dest="activation",\
                        help="Activation function",choices=['relu','sigmoid','tanh'])
    parser.add_argument('--tune', action='store_true',\
                        help='Tune the model')

    return parser.parse_args(sys.argv[1:])

def load_data(train=0):
#    first way to load data
    images = []
    dict = {}
    dict[1]={}
    dict[1]['label'] = os.path.join(DATA_PATH, "train-labels-idx1-ubyte.gz")
    dict[1]['image'] = os.path.join(DATA_PATH, "train-images-idx3-ubyte.gz")
    dict[0]={}
    dict[0]['label'] = os.path.join(DATA_PATH, "t10k-labels-idx1-ubyte.gz")
    dict[0]['image'] = os.path.join(DATA_PATH, "t10k-images-idx3-ubyte.gz")
    with gzip.open(dict[train]['label'],'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        labels = array("B", file.read())
    with gzip.open(dict[train]['image'], 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        image_data = array("B", file.read())

    [images.append([0] * rows * cols) for i in range(size)]

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    image = np.ndarray((len(images),28,28,1))

    for i in range(len(images)):
        image[i,:] = np.reshape(images[i],(28,28,1))

    return image, np.transpose(np.matrix(labels))

# =============================================================================
# #   second way to load data - deprecated
#     # training images
#     with open(os.path.join(DATA_PATH, "train-images-idx3-ubyte.gz"), "rb") as f:
#         train_images = extract_images(f)
#
#     # training labels
#     with open(os.path.join(DATA_PATH, "train-labels-idx1-ubyte.gz"), "rb") as f:
#         train_labels = extract_labels(f)
#
#     # testing images
#     with open(os.path.join(DATA_PATH, "t10k-images-idx3-ubyte.gz"), "rb") as f:
#         test_images = extract_images(f)
#
#     # testing labels
#     with open(os.path.join(DATA_PATH, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
#         test_labels = extract_labels(f)
#
#     return (train_images, train_labels), (test_images, test_labels)
# =============================================================================

# =============================================================================
# #    third way to load data - using tensorflow
#     import tensorflow as tf
#     old_v = tf.logging.get_verbosity()
#     tf.logging.set_verbosity(tf.logging.ERROR)
#
#     from tensorflow.examples.tutorials.mnist import input_data
#
#     mnist = input_data.read_data_sets("../dataset/", one_hot=True)
#     train_data = mnist.train.images  # Returns np.array
#     train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#     eval_data = mnist.test.images  # Returns np.array
#     eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#     tf.logging.set_verbosity(old_v)
#     return train_data, train_labels, eval_data, eval_labels
# =============================================================================

class main():
    def __init__(self):
        self.train_img, self.train_lbl = load_data(1)

        self.input_shape = (28,28,1)
        # Hyper parameter values obtained after tuning using gird search
        self.eta = 1e-3
        self.epochs = 20
        self.batch_size = 64
        self.kernel_size=(4,4)
        self.model = CNN()

    def train(self,activation):
        history={}
        if activation is None:
            activation=[]
            activation.append('relu')

        for act in activation:
            print("\nTraining for {} activation".format(act))
            self.model.create_classifier(input_shape=self.input_shape,activation=act,eta=self.eta, kernel_size=self.kernel_size)

            history[act] = self.model.train_classifier(self.train_img, self.train_lbl,epochs=self.epochs,batch_size=self.batch_size, activation=act)

            #summarize history for accuracy
            plt.figure(str(act)+' acc')
            plt.plot(history[act].history['acc'])
            plt.plot(history[act].history['val_acc'])
            plt.title(str(act)+' model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='lower right')
            plt.savefig(str(act)+'_accuracy.png')

            # summarize history for loss
            plt.figure(str(act)+' loss')
            plt.plot(history[act].history['loss'])
            plt.plot(history[act].history['val_loss'])
            plt.title(str(act)+' model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.savefig(str(act)+'_loss.png')

        legend=[]

        for values in history:
            plt.figure('accuracy')
            plt.plot(history[values].history['acc'])
            legend.append(values)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(legend, loc='lower right')
        plt.title('Accuracies for different activations')
        plt.savefig('combined train accuracy.png')

        for values in history:
            plt.figure('val accuracy')
            plt.plot(history[values].history['val_acc'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(legend, loc='lower right')
        plt.title('Validation Accuracies for different activations')
        plt.savefig('combined val accuracy.png')

        for values in history:
            plt.figure('loss')
            plt.plot(history[values].history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(legend, loc='upper right')
        plt.title('Loss for different activations')
        plt.savefig('combined train loss.png')

        for values in history:
            plt.figure('val loss')
            plt.plot(history[values].history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(legend, loc='upper right')
        plt.title('Validation loss for different activations')
        plt.savefig('combined val loss.png')

    def tune(self):
            self.model.create_classifier(input_shape=self.input_shape,eta=self.eta, kernel_size=self.kernel_size)
            grid_result = self.model.tune_hyperparameters(self.train_img, self.train_lbl)

    def test(self,activation):
        if activation is None:
            activation=[]
            activation.append('relu')
        test_img, test_lbl = load_data(0)
        for act in activation:
            print("Test for "+act+":")
            test_accuracy, test_loss, predictions = self.model.test_classifier(test_img,test_lbl,act)

            print("\n")
            predictions = np.round(predictions)
            predictions = predictions.astype(int)
            df = pd.DataFrame(predictions)
            df.to_csv("mnist.csv", header=None, index=None)


if __name__ == '__main__':
    FLAGS = get_args()
    main = main()
    if FLAGS.train:
#        if FLAGS.activation is not None:
        main.train(FLAGS.activation)
#        else:
#            print('Pass activation argument/s for training')
    if FLAGS.test:
#        if FLAGS.activation is not None:
        main.test(FLAGS.activation)
#        else:
#            print('Pass activation argument/s for testing')
    if FLAGS.tune:
        main.tune()
#    if FLAGS.load:
#        train_img, train_lbl = load_data()
