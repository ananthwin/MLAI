#Neural net :Bunch of node and neuroan are connected rember number 9, it has 3
#layers 1st layer(info layer),2nd hidden layer and 3nd output layer
#Keras: Allow defined diferent gray sturcure utalizign sequental of vertial
#columnet in nodes
#Currentlying runnin gin 3.7
import tensorflow as tf
import numpy as np
from tensorflow import keras

#Printing stuff
import matplotlib.pyplot as plt 

# Load a pre-defined dataset(70k of 28*28)
fashion_mnist = keras.datasets.fashion_mnist

#Pull out odata from dataset
(train_images,train_lables),(test_images,test_lables) = fashion_mnist.load_data()

#Show data
#print(train_lables[0])
#print(train_images[0])
#plt.imshow(train_images[0], cmap='gray',vmin=0,vmax=255)
#plt.show()

#Defin our neural net structure
model = keras.Sequential([#Input is a 28*28 image("Flatten" flatten the 28*28 into a single 784*1
    #input layer)
    #Why we need to flattern: To simplfy the strucure of the nueral net singler
    #coloumn, put pixel in one long layer
    keras.layers.Flatten(input_shape=(28,28)),

    #hidden layer in 128 deep.  return the value or 0 works good enough much
    #why we need to hidden layer:
    #faster)
    #keras.layers.Dense(units=128, activation=tf.nn.relu),

    #Output is 0-10 (depending on what piece of clothing it is).  return
    #maximum,Dense means:
    keras.layers.Dense(units=10,activation=tf.nn.softmax)])

#Compile our model
#loss how wrong we are
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train our model, using our training data epoch means how many time it
#can we increase epoch times(ex 10)
model.fit(train_images, train_lables,epochs=5)

#Train our mode, using our traning
test_loss = model.evaluate(test_images,test_lables)

plt.imshow(test_images[1], cmap='gray',vmin=0,vmax=255)
plt.show()

print(train_lables[1])

#Make preditions
predictions=model.predict(test_images)
print(predictions[1])

#Print out prediction
print(list(predictions[1].index(max(predictions[1]))))
print('code completed')