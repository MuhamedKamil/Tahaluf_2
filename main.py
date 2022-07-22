import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt
from net_flops import net_flops

BATCH_SIZE = 256
EPOCHS = 5
LearningRate = 0.001
img_height = 28
img_width = 28
input_shape = (28, 28, 1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
checkpoint_path = "training_1/cp.ckpt"
num_compute_expensive_layers = 2


# -----------------------------------------------------------------------------------------------------------------------
def load_normalize_data():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, Y_train), (X_test, Y_test)


def build_graph(shape):
    inp = Input(shape=shape, name='input_image')
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='feature_grid')(
        inp)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(rate=0.25)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    model = Model(inp, x)
    model.load_weights(checkpoint_path)
    return model


def train(training_samples, training_labels):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    model = build_graph(input_shape)
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(LearningRate), metrics=['accuracy'])
    model.fit(training_samples, training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2,
              callbacks=[cp_callback])


def get_complex_layers(layers, number):
    sorted_Layers = sorted(layers.items(), key=lambda x: x[1], reverse=True)
    return sorted_Layers[:number]


# -----------------------------------------------------------------------------------------------------------------------
# Receptive field of the model
(X_train, Y_train), (X_test, Y_test) = load_normalize_data()
ReceptiveField = KerasReceptiveField(build_graph)
ReceptiveField_params = ReceptiveField.compute(input_shape, 'input_image', ['feature_grid'])
print(ReceptiveField_params)
ReceptiveField.plot_rf_grids(X_train[1111], figsize=(28, 28))
plt.show()
# -----------------------------------------------------------------------------------------------------------------------
# Calculate number of FLOPS and MACCs
model = build_graph(input_shape)
flops = net_flops(model, table=True)
# expensive_layers = get_complex_layers(flops, num_compute_expensive_layers)
# print(type(expensive_layers))
