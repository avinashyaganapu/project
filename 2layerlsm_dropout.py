# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 06:32:49 2018

@author: avina
"""

import re
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import tensorflow as tf

import csv
final=pd.read_csv('final.csv')

with open('total.csv', 'rt', encoding="utf8") as t:
    reader = csv.reader(t)
    total=list(reader)
    

fd = final['Score'].tolist()

#splitting train and test data in the ratio 80:20
x_train, x_test, y_train, y_test = train_test_split(total,fd,test_size=0.2,shuffle=False)

print("-----------------------TRAIN DATA------------------------------------")
print(len(x_train))
print(len(y_train))
print("---------------------------------------------------------------------")
print("\n-----------------------TEST DATA-------------------------------------")
print(len(x_test))
print(len(y_test))

#padding
X_train = sequence.pad_sequences(x_train, maxlen = 700)
X_test = sequence.pad_sequences(x_test, maxlen = 700)

print("-----------------------TRAIN DATA------------------------------------")
print(X_train.shape)
print(len(y_train))
print("---------------------------------------------------------------------")
print("\n-----------------------TEST DATA-------------------------------------")
print(X_test.shape)
print(len(y_test))

#model
model = Sequential()
model.add(Embedding(41500, 32, input_length = 700))
model.add(Dropout(0.20))
model.add(LSTM(100,return_sequences=True))
model.add(Dropout(0.20))
model.add(LSTM(100))
model.add(Dropout(0.20))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#fitting the model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

#plotting train vs val loss
def Plot(err):
    x = list(range(1,11))
    v_loss = err.history['val_loss']
    t_loss = err.history['loss']
    plt.plot(x, v_loss, '-b', label='Validation Loss')
    plt.plot(x, t_loss, '-r', label='Training Loss')
    plt.legend(loc='center right')
    plt.xlabel("EPOCHS",fontsize=15, color='black')
    plt.ylabel("Train Loss & Validation Loss",fontsize=15, color='black')
    plt.title("Train vs Validation Loss on Epoch's" ,fontsize=15, color='black')
    plt.show()
Plot(history)

#calculating accuracy 
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))