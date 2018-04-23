#MNIST v1
#First Model
""""This Model got 94.9% training set accuracy"""
"""Changes :
    
    * Added mini batch gradient descent
    * Mini batch size = 80    
        
    """




import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mnist = pd.read_csv('train_mnist.csv')
X_train = mnist.iloc[:,1:].values
Y_train = mnist.iloc[:,0].values
X_train = X_train.reshape(-1,28,28,1)
Y_train = Y_train.reshape(Y_train.shape[0],1)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categorical_features=[0])
Y_train = encoder.fit_transform(Y_train).toarray()



#Placeholders
def create_placeholders(n_H, n_W, n_C, n_y):
    
    X = tf.placeholder(tf.float32, [None, n_H, n_W, n_C])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


#Filters
def initialize_parameters():
    
    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))    
    
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters


#Forward prop
def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 4, 4, 1], strides=[1 , 4, 4, 1], padding='SAME')
    
    
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1 , 2, 2, 1], padding='SAME')
    
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2 , 10, activation_fn=None)
    
    return Z3


#COST FUNCTION
def compute_cost(Z3, Y):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3))
    return cost
    




"""FULL MODEL"""
tf.reset_default_graph()
(m, n_H, n_W, n_C) = X_train.shape 
n_y = Y_train.shape[1]

costs = []

X, Y = create_placeholders(n_H, n_W, n_C, n_y)

parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)
cost = compute_cost(Z3, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.009).minimize(cost)
init = tf.global_variables_initializer()


#For mini Batches
permuts = list(np.random.permutation(m))
minibatch_size = 80
shuffeled_X = X_train[permuts, :,:,:]
shuffeled_Y = Y_train[permuts, :]

with tf.Session() as sess:     
    sess.run(init)
    for epoch in range(10):
        
        num_minibatches = int(m/minibatch_size)
        
        for i in range(num_minibatches):
            mb_X = shuffeled_X[i*minibatch_size:(i+1)*minibatch_size, :, :, :]
            mb_Y = shuffeled_Y[i*minibatch_size:(i+1)*minibatch_size, :]
            
            
            _ , mb_cost = sess.run([optimizer, cost], feed_dict={X: mb_X, Y:mb_Y})
            costs.append(mb_cost)
    


        matches = tf.equal(tf.argmax(Z3,1),tf.argmax(Y,1))
        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        print("Train Accuracy after " + str(epoch) + " epochs")
        print(sess.run(acc,feed_dict={X:X_train,Y:Y_train}))









