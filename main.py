import cv2 as cv  # needed to import images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the dataset of the handwritten digits
mnist = tf.keras.datasets.mnist

# split data to training data and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale down the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Defining the model. one input layer, two hidden layers and one output layer
model = tf.keras.models.Sequential()  # feed forward neural network

# the input layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# the next layers (Hidden layers)
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

# the output layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
model.save('digits.model')

# loading our images to classify them with the neural network
for x in range(10):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
    img = np.invert(np.array([img]))

    prediction = model.predict(img)

    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
