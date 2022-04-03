import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import random

def generator(Z,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)

    return out

def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)

    return out, h3

X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step

for i in range(100001):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
'''
def GAN_model(attributes, classifier, holdout_attributes, holdout_classifiers, input_dim=190, subject_count=60, artificial_count=0):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    
    # model.add(layers.Embedding(input_dim=input_dim, output_dim=190))

    # # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    # model.add(layers.GRU(10, return_sequences=True))

    # # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    # model.add(layers.SimpleRNN(64))

    model.add(layers.Dense(64, activation="softsign"))
    # model.add(layers.LSTM(32))
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
    '''