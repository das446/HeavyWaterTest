from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.optimizers import SGD
import sys

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(threshold=sys.maxsize)

def ReadFile(file):
    Labels = np.array([])
    Numbers = []
    #data = [[],[]] #labels, numbers
    with open(file) as f:
        lines = f.read().splitlines()
        
        for i, line in enumerate(lines):
            label = line.split(',')[0]
            numbers = line.split(',')[1]
            numbers = numbers.split(' ')
            numbers = list(filter(lambda x: x!='', numbers))
            numbers = list(map(lambda x: int(x,16), numbers))
            Labels = np.append(Labels,label)
            Numbers.append(numbers)
            
    Numbers = NormalizeData(Numbers)
    Numbers = np.array(Numbers)
    
    data = [Numbers, Labels]
    return data

#makes every list the same length by appending 0s to the end
def NormalizeData(data):
    longest = LongestList(data)
    for line in data:
        l = len(line)
        if l < longest:
            dif = longest - l
            for x in range(0,dif):
                line.append(0)
    return data

def LongestList(lst):
    if type(lst) is not list:
        return 0
    max = len(lst)
    for i in lst:
        max_i = LongestList(i)
        if max_i > max:
            max = max_i
    return max

def Encode(data):
    values = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #encoded = keras.utils.to_categorical(integer_encoded)
    return integer_encoded









data = ReadFile("shuffled-full-set-hashed.csv")
#predict = ReadFile("predict.csv")
l = len(data[0]/2)
x_train = data[0][:len]
y_train = data[1][:len]


#predict = ReadFile("predict.csv")
x_test = data[0][len:]
y_test = data[1][len:]

size = len(y_train)
width = len(x_test[0])
one_hot_labels = Encode(y_train)
one_hot_labels_train = Encode(y_test)

print("size=",size,", width=",width)

model = Sequential()
model.add(Dense(units=32, input_dim=size, activation='relu'))
model.add(Dense(units=size, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())

score = model.fit(x_train, one_hot_labels, verbose=1, batch_size=128, epochs=6)
#print(score)
score = model.evaluate(x_test, one_hot_labels_train, batch_size=128)




