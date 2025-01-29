import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from foodscore import params
import os




class RandomBlurHue(tf.keras.layers.Layer):

    def __init__(self, kernel_size=5, sigma_max=3.0, hue_max_delta=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.hue_max_delta = hue_max_delta

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "sigma_max": self.sigma_max,
            "hue_max_delta": self.hue_max_delta,
        })
        return config

    def call(self, inputs, training):
        if not training:
            return inputs

        outputs = tf.image.random_hue(inputs, self.hue_max_delta)
        sigma = np.random.uniform(0.0, self.sigma_max)
        outputs = tfa.image.gaussian_filter2d(
            outputs,
            filter_shape=(self.kernel_size, self.kernel_size),
            sigma=sigma,
            name="Gaussian-blur",
            padding="REFLECT",
        )

        return outputs

## Create the model
def create_model(input_shape = params.INPUT_SHAPE):

    inputs = layers.Input(input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomBrightness(0.3)(x)
    x = layers.RandomContrast(0.3)(x)
    x = layers.RandomZoom((-0.3, 0))(x)
    x = RandomBlurHue(kernel_size=5, sigma_max=2.0)(x)

    x = preprocess_input(x)

    # base model
    resnet = ResNet152(include_top=False, weights='imagenet')

    x = resnet(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="softmax", name="class_label", kernel_regularizer=l2(0.01))(x)

    # Freeze ResNet weights
    resnet.trainable = False

    return Model(inputs=inputs, outputs=[x])

## Fit the model

def fit_model(model, train_ds, val_ds):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=90,
    decay_rate=0.8)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=lr_schedule),
        metrics="categorical_accuracy",
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        verbose = 0,
        restore_best_weights = True
    )

    fit_history = model.fit(
        train_ds,
        validation_data=(val_ds),
        batch_size=128,
        epochs=1,
        verbose=1,
        callbacks = [early_stopping],
    )

    return fit_history

## load saved model

def load_model(path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),params.MODEL_PATH)):
    saved_model = models.load_model(path)
    return saved_model

## predict label

def predict_label(saved_model, image, most_prob=5):
    pred = saved_model.predict(image)
    food_list = []
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'raw_data/UECFOOD100/category.txt'), 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            food_list.append(line[1])
    data = list(zip(food_list, pred[0]))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    result = [x[0] for x in sorted_data[:most_prob]]
    return result
