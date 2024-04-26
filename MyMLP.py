import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

from sklearn.metrics import accuracy_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

##############################################
#
# 2 layer perceptron for classification
#
##############################################
class MyMLP():
    # Parameters:
    # hidden_size - dimension of hidden layer
    # max_epochs - number of max epochs
    # dropout_rate - drop out rate
    # l2_reg - regularization
    # loss - loss function ('binary_crossentropy' for binary classification)
    # optim - optimization function
    def __init__(self, hidden_size, max_epochs, dropout_rate=0.2, l2_reg=0.1, loss='binary_crossentropy', optim='adagrad'):
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.optim = optim
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.mlp = Sequential()
        
        # define layers
        self.mlp.add(Dense(self.hidden_size, activation='relu'))
        # self.mlp.add(Dense(self.hidden_size, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg)))
        # self.mlp.add(Dropout(self.dropout_rate))
        self.mlp.add(Dense(1, activation='sigmoid'))
        
    def train(self, train_readout, train_target):
        # set up optimizer and loss function
        self.mlp.compile(optimizer=self.optim, loss=self.loss, metrics=['accuracy'])
        
        # fit data
        history = self.mlp.fit(train_readout, train_target, batch_size = 0, epochs=self.max_epochs, verbose=0)
        
        # # print information during training
        # for epoch, loss in enumerate(history.history['loss']):
        #     accuracy = history.history['accuracy'][epoch]
        #     print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {loss}, Accuracy: {accuracy}")

        # print final training loss and accuracy
        final_loss, final_accuracy = history.history['loss'][-1], history.history['accuracy'][-1]
        print(f"Final Loss: {final_loss}, Final Accuracy: {final_accuracy}")
            
        # return final loss
        return final_loss

    
    def test(self, test_readout, test_target):
        predictions = self.mlp.predict(test_readout)

        # round preds for binary classification
        predicted_labels = np.round(predictions).astype(int)

        # calc accuracy
        accuracy = accuracy_score(test_target, predicted_labels)
        print("Test Accuracy: ", accuracy)
        
        return predicted_labels, accuracy