#MNIST v1
#First Model
""""This Model got 99.09% training set accuracy after 20 epochs"""
"""Changes :
    
    * Changed NN structure
    * Mini batch size = 50    
    * Added biases
    * Reduced learning rate from 0.009 to 0.001
    
    * Added more fully connected layers
    """


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


mnist = pd.read_csv('train_mnist.csv')
X_train = mnist.iloc[:,1:].values
Y_train = mnist.iloc[:,0].values
X_train = X_train.reshape(-1,28,28,1)
Y_train = Y_train.reshape(Y_train.shape[0],1)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categorical_features=[0])
Y_train = encoder.fit_transform(Y_train).toarray()



#Placeholders
def create_placeholders(n_H, n_W, n_C, n_Y):
    
    X = tf.placeholder(tf.float32, shape= [None, n_H, n_W, n_C])
    Y = tf.placeholder(tf.float32, shape= [None, n_Y])
    
    return X, Y




#Variables or Weights or filters
def initialize_parameters():
    
    W1 = tf.get_variable('W1', shape=[3, 3, 1, 8], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable('W2', shape=[6, 6, 8, 16], initializer= tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[8], initializer= tf.zeros_initializer())
    b2 = tf.get_variable('b2', shape=[16], initializer= tf.zeros_initializer())
    
    parameters = {'W1': W1,
                  'W2': W2,
                  'b1': b1,
                  'b2': b2
                  }
    
    return parameters




#Forward Propagation
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    Z1 = tf.nn.conv2d(X, W1, strides= [1, 1, 1, 1],padding='SAME')
    A1 = tf.nn.relu(Z1 + b1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 3, 3, 1], strides= [1, 3, 3, 1], padding='SAME')
    
    
    Z2 = tf.nn.conv2d(P1, W2, strides= [1, 1, 1, 1],padding='SAME')
    A2 = tf.nn.relu(Z2 + b2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 6, 6, 1], strides= [1, 6, 6, 1], padding='SAME')
    
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 20)
    
    Z4 = tf.contrib.layers.fully_connected(Z3, 20)
    Z5 = tf.contrib.layers.fully_connected(Z4, 10, activation_fn=None)
    
    
    return Z5



#Cost Function
def compute_cost(Z3, Y):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3))
    return cost




tf.reset_default_graph()

(m, n_H, n_W, n_C) = X_train.shape
n_Y = Y_train.shape[1]

costs = []

X, Y = create_placeholders(n_H, n_W, n_C, n_Y)
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
cost = compute_cost(Z3, Y)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
init = tf.global_variables_initializer()


#Preparing mini batches
permuts = list(np.random.permutation(m))
X_shuffle = X_train[permuts, :, :, :]
Y_shuffle = Y_train[permuts, :]


minibatch_size = 50
num_epochs = 20

with tf.Session() as sess:
    sess.run(init)    
    for epoch in range(num_epochs):        
        num_minibatches = int(m/minibatch_size)
        
        for k in range(num_minibatches):
            
            X_batch = X_shuffle[k*minibatch_size:(k+1)*minibatch_size, :, :, :]
            Y_batch = Y_shuffle[k*minibatch_size:(k+1)*minibatch_size, :]
            
            
            _, mb_cost = sess.run([optimizer, cost], feed_dict={X: X_batch, Y:Y_batch})
            costs.append(mb_cost)
            
        
        matches = tf.equal(tf.argmax(Y, 1), tf.argmax(Z3, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))        
        print("Train Accuracy after " + str(epoch) + " epochs")
        print(sess.run(acc,feed_dict={X:X_train,Y:Y_train}))

            
        
    




















