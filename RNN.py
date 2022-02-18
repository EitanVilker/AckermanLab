import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import random

def RNN_model(attributes, classifier, holdout_attributes, holdout_classifiers, input_dim=190, subject_count=60, artificial_count=0):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    
    # model.add(layers.Embedding(input_dim=input_dim, output_dim=190))

    # # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    # model.add(layers.GRU(10, return_sequences=True))

    # # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    # model.add(layers.SimpleRNN(64))

    model.add(layers.Dense(64, activation="softsign"))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(3, activation='sigmoid'))

    model.summary()

    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="sgd",
    metrics=["accuracy"],
    )

    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=pd.DataFrame(attributes).max(), epochs=1)
    model.fit(attributes, classifier, validation_data=(holdout_attributes, holdout_classifiers), batch_size=64, epochs=200)
    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=64, epochs=200)