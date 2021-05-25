import tensorflow_hub as hub
import seaborn
import os
import re
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
import datetime
import time
import csv
from keras.models import load_model
from keras import Sequential
from keras import layers
from keras import Model
from keras.layers import LSTM, TimeDistributed, Dense, GlobalMaxPooling1D
from keras.optimizers import RMSprop
import numpy as np
import pylab as pl 
import matplotlib.pyplot as plt
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
    	signature="default", as_dict=True)["default"]
tf.disable_eager_execution()
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/2")
data = pd.read_csv('dialogue1.csv')
data = data[['text', 'Label']]
data.columns =  ['text', 'label']
data_test = pd.read_csv('test_file.csv')
data_test = data_test.dropna(how='any',axis=0)
df = pd.DataFrame(data, columns=['text', 'label'])
df_test = pd.DataFrame(data_test, columns=['Utterance'])
df_train=df
df.label = df.label.astype('category')
train_text = df_train['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = np.asarray(pd.get_dummies(df_train.label), dtype = np.int8)
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding,
            output_shape=(512,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
dense1 = layers.Dense(128, activation='relu')(dense)
pred = layers.Dense(41, activation='softmax')(dense1)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', 
optimizer='adam', metrics=['accuracy'])
new_text=df_test.Utterance
new_text = np.array(new_text, dtype=object)[:, np.newaxis]
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./model.h6')  
    predicts = model.predict(new_text, batch_size=32)
categories = df_train.label.cat.categories.tolist()
predict_logits = predicts.argmax(axis=1)
predict_labels = [categories[logit] for logit in predict_logits]
predict_labels=str(predict_labels)
predict_labels=predict_labels.replace("['","")
predict_labels=predict_labels.replace(",","")
predict_labels=predict_labels.replace("']","")
predict_labels=predict_labels.replace("'","")
ff=open('result.File','a')
ff.write(str(predict_labels))
ff.close()
