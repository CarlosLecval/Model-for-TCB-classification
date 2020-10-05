import pandas as pd
import numpy as np
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.utils import plot_model
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load pre-processed images and results
x = np.load('preprocessed_img.npy')
y=np.load('results.npy')

#split images for training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=2020)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=2020)

#apply standard scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)

#define a model
modelo = Sequential()
modelo.add(Dense(2048,input_dim=10000,kernel_initializer='normal',kernel_regularizer=regularizers.l1_l2(0.01), activation='relu'))
modelo.add(Dense(1024,kernel_regularizer=regularizers.l1(0.01),activation='relu'))
modelo.add(Dense(512,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
modelo.add(Dense(1,activation='sigmoid'))
modelo.compile(optimizer=Adam (learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True,) 
                ,loss='binary_crossentropy',metrics=['accuracy'])
#train model
historia = modelo.fit(x_train_scaled,y_train,validation_data=(x_val_scaled,y_val),batch_size=16,epochs=10)

#metrics
plt.plot(historia.history['loss'])
plt.plot(historia.history['val_loss'])
plt.show()
plt.plot(historia.history['accuracy'])
plt.plot(historia.history['val_accuracy'])
plt.show()