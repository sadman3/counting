# -*- coding: utf-8 -*-
"""counting_tf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10yhlmiksbKMMTGvACrKp8lEgF8WNtxxO
"""

import tensorflow as tf
from utils import LoadData, DrawGraph


def PretrainedModel():
    new_input = tf.keras.Input(shape=(186, 116, 3))
    pre_trained_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, input_tensor=new_input, pooling='avg')

    #pre_trained_model.trainable = False

    # # # custom modifications on top of pre-trained model
    model = tf.keras.models.Sequential()
    model.add(pre_trained_model)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(
        # set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
        optimizer=tf.keras.optimizers.Adam(0.001),
        # sparce categorical cross entropy (measure predicted dist vs. actual)
        loss=tf.keras.losses.MeanSquaredError(),
        # how often do predictions match labels
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model


def CustomModel():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(186, 116, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # model.add(tf.keras.layers.Conv2D(
    #     32, (3, 3), activation='relu', input_shape=(186, 116, 3)))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dense(1))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        # how often do predictions match labels
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


def Train(checkpoint_name):
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./models/"+checkpoint_name+".h5",
                                                     verbose=1,
                                                     save_best_only=True,
                                                     monitor='val_loss',
                                                     mode='min'),
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.001, patience=10)]

    trainingData, trainingLabels, validationData, validationLabels = LoadData(
        "train")
    model = CustomModel()
    #model = PretrainedModel()
    print("Model created.")
    history = model.fit(trainingData, trainingLabels, epochs=100, batch_size=32, validation_data=(
        validationData, validationLabels), callbacks=checkpoint)

    DrawGraph(history.history['loss'], history.history['val_loss'])
    print("Model saved.")


if __name__ == '__main__':
    Train("checkpoint")
