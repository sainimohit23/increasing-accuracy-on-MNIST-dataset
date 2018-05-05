#Very basic 1 layered keras model
#Model got 99.51 train accuracy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras
import keras.backend as k


mnist = pd.read_csv('train_mnist.csv')
X_train = mnist.iloc[:,1:].values
Y_train = mnist.iloc[:,0].values



#DATA PREPROCESSING
from sklearn.preprocessing import OneHotEncoder
Y_train = Y_train.reshape(Y_train.shape[0],1)
encoder = OneHotEncoder(categorical_features=[0])
Y_train = encoder.fit_transform(Y_train).toarray()



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(-1,28,28,1)


def mnist_model(input_shape):
    
    X_input = keras.layers.Input(input_shape)
    
    X = keras.layers.ZeroPadding2D((2, 2))(X_input)
    X = keras.layers.Conv2D(16, (5, 5), name='conv0')(X)
    X = keras.layers.BatchNormalization(axis=3, name= 'Bn0')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.MaxPooling2D((2, 2), name='max_pool1')(X)
    
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(10, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.models.Model(inputs = X_input, outputs = X, name='HappyModel')
   
    return model    


mymodel = mnist_model(X_train.shape[1:])
mymodel.compile(optimizer="Adam", loss="binary_crossentropy", metrics = ["accuracy"])
mymodel.fit(x=X_train, y=Y_train, epochs=5, batch_size=64)


preds = mymodel.evaluate(x=X_train, y=Y_train)

print()
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))

































