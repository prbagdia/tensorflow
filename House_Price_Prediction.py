# Prediction of House prices, for now the only parameter being included here is size of the house

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation # support for animation

# generate random house sizes between 1000 and 3500 (say in sq feet)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low = 1000, high = 3500, size = num_house)

#generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low = 20000, high = 70000, size = num_house)


# Plot the house size and their prices

plt.plot(house_size,house_price,"bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()


#normalize values to prevent under/overflows
def normalize(array):
    return (array-array.mean())/array.std()

# split the data into testing and training samples -> 70% for training and remaining for testing
num_train_samples = math.floor(num_house*0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define testing data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

#setup tensorflow placeholders that get upadted as we descend down the gradient
tf_house_size = tf.placeholder("float",name="house_size")
tf_price = tf.placeholder("float",name="price")

# define the varibales holding the size_factor and price_offset we set during training
# initialize them to a random normal value
tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")


# define the price_prediction
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

#define the loss function (Mean squared error here)
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price,2))/(2*num_train_samples)

#Optimizer learning rate
learning_rate= 0.1

#Defie a gradient descent optimizer that will minimize the loss defined in the operation 'Cost'
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


#initialize the variables
init = tf.global_variables_initializer()

#launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    #set how often to display trainning progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    # keep iterating the training data
    for iteration in range(num_training_iter):

        #For all training data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict = {tf_house_size : x, tf_price : y})

        #Display the current status
        if (iteration+1)%display_every == 0:
            c=sess.run(tf_cost,feed_dict= {tf_house_size : train_house_size_norm, tf_price : train_price_norm})
            print("Iteration #:",'%04d' % (iteration+1), "cost= ", "{:.9f}".format(c),\
                "size_factor=", sess.run(tf_size_factor), "price_offset= ", sess.run(tf_price_offset))

    print("Optimization finished!")
    training_cost = sess.run(tf_cost,feed_dict = {tf_house_size : train_house_size_norm, tf_price : train_price_norm})
    print("Trained cost=", training_cost,"size_factor=", sess.run(tf_size_factor),"price_offset=", sess.run(tf_price_offset), "\n")

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()
    

    #Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")

    plt.plot(train_house_size,train_price,'go',label='Training Data')
    plt.plot(test_house_size,test_price,'mo',label='Testing Data')
    plt.plot(train_house_size_norm*train_house_size_std+train_house_size_mean, 
         (sess.run(tf_size_factor)*train_house_size_norm + sess.run(tf_price_offset))* train_price_std + train_price_mean,
         label = "Learned Regression")

    plt.legend(loc="upper left")
    plt.show()
     














