# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 01:35:54 2019

@author: Shalin
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

#from keras.models import model_from_json
from sklearn.model_selection import train_test_split

class CNN(object):
    def __init__(self):
        pass

    def create_classifier(self,input_shape=(28,28,1),activation=None, eta=1e-3, kernel_size=(4,4)):
        # Initialising the CNN
        self.classifier = Sequential()

        # Step 1 - Convolution
        self.classifier.add(Conv2D(16, kernel_size=kernel_size, input_shape = input_shape, activation = activation))

        self.classifier.add(BatchNormalization())

        # Step 2 - Pooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Dropout(0.25))

        # Adding a second convolutional layer
        self.classifier.add(Conv2D(32, (3, 3), activation = activation))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        self.classifier.add(Flatten())

        # Step 4 - Full connection
        self.classifier.add(Dense(units = 128, activation=activation))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units = 10, activation='softmax'))

        # Compiling the CNN
        optimizer=Adam(lr=eta)
        self.classifier.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.classifier

    def tune_hyperparameters(self, train_images, train_labels):
        epochs = [10, 15,20]
        model = KerasClassifier(build_fn=self.create_classifier)
        batch_size = [32, 64, 128]
        learning_rate = [1e-2, 1e-3, 1e-4]
        kernel_size = [(2,2),(3,3), (4,4)]
        param_grid = dict(batch_size=batch_size, eta=learning_rate, kernel_size=kernel_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
        grid_result = grid.fit(train_images, train_labels)
        return grid_result

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        return grid_result

    def train_classifier(self, train_images, train_labels, batch_size=256, epochs=10,activation="relu"):
        history = self.classifier.fit(train_images, train_labels, batch_size, epochs, validation_split=0.25)
        self.classifier.save_weights(str('model'+str(activation)))
        return history

    def split_data(self, train_images, train_labels):
        train_img, train_lbl, val_img, val_lbl = train_test_split(train_images, train_labels, test_size = 0.25)
        return train_img, train_lbl, val_img, val_lbl

    def test_classifier(self,test_images,test_labels,activation):
        self.classifier = self.create_classifier(input_shape=(28,28,1), activation=activation, eta=1e-3)
        self.classifier.load_weights(str('model'+str(activation)))

        loss, acc = self.classifier.evaluate(test_images, test_labels,verbose=0)
        print("Loss = {}, accuracy = {}".format(loss, acc))
        predictions = self.classifier.predict_proba(x=test_images, batch_size=None, verbose=0)
        return acc, loss, predictions
