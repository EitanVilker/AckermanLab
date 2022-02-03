import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import random

def RNN_model(attributes, classifier, input_dim=190, subject_count=60):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    # model.add(layers.Embedding(input_dim=input_dim, output_dim=190))

    # # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    # model.add(layers.GRU(10, return_sequences=True))

    # # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    # model.add(layers.SimpleRNN(64))

    # model.add(layers.Dense(64))
    model.add(layers.Dense(14, activation='sigmoid'))

    model.summary()

    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
    )

    holdout_attributes = []
    holdout_classifiers = []
    to_remove = []
    for i in range(subject_count // 10):
        # Choose a random subject index that hasn't been chosen yet
        rand = random.randint(0, subject_count - i - 1)
        while rand in to_remove:
            rand = random.randint(0, subject_count - i - 1)
        to_remove.append(rand)
        
        holdout_attributes.append(attributes.loc[rand])
        holdout_classifiers.append(classifier.loc[rand])
        classifier.pop(rand)

    print(to_remove)
    attributes = attributes.drop(to_remove)
    holdout_attributes = np.asarray(holdout_attributes)
    holdout_classifiers = np.asarray(holdout_classifiers)

    # model.fit(attributes, classifier, validation_data=(attributes, classifier), batch_size=pd.DataFrame(attributes).max(), epochs=1)
    model.fit(attributes, classifier, validation_data=(holdout_attributes, holdout_classifiers), batch_size=64, epochs=200)