import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import random

def NN_model(attributes, classifier, holdout_attributes, holdout_classifiers, input_dim=191, subject_count=60, artificial_count=0):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))

    model.add(layers.Dense(64, activation="softsign"))
    model.add(layers.Dense(14, activation='sigmoid'))

    model.summary()

    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="sgd",
    metrics=["accuracy"],
    )

    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=pd.DataFrame(attributes).max(), epochs=1)
    model.fit(np.asarray(attributes).astype('float32'), np.asarray(classifier).astype('float32'), validation_data=(np.asarray(holdout_attributes).astype('float32'), np.asarray(holdout_classifiers).astype('float32')), batch_size=64, epochs=200)
    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=64, epochs=200)