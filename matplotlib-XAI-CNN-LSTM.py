from tensorflow.keras.layers import *
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline
from numpy.random import seed
#%tensorflow_version 1.x

#import and read data
data = r"C:\Users\hp\Desktop\priyasoftweb\STM32\export_date.csv"

merged_data =pd.read_csv(data)
merged_data =merged_data.iloc[:, :-1]

print("Dataset shape:", merged_data.shape)
merged_data.head()


train = merged_data.iloc[:35243, :]
test = merged_data.iloc[35243:, :]

print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)


# normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train)
X_test_scaled = scaler.transform(test)

#scaled the data
X_train = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
print("Test data shape:", X_test.shape)

from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Input, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from tensorflow.keras import regularizers

#stacked lstm and conv1d with 98% accuracy
from tensorflow.keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from tensorflow.keras.models import Model
import datetime
def autoencoder_model(X):
    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(input_layer)
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    output_layer = TimeDistributed(Dense(X.shape[2]))(lstm1)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

model = autoencoder_model(X_train)
model.compile(optimizer="adam", loss='mae', metrics=["accuracy", 
          "mean_squared_error", "mean_absolute_error"])

model.summary()

#from sklearn.metrics import  make_scorer, accuracy_score, confusion_matrix, classification_report
#from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, roc_curve, precision_score
#from sklearn.metrics import auc, recall_score, r2_score, mean_absolute_error    

#fit
nb_epochs = 100
batch_size =round(len(X_train)/7)

import datetime
t_ini = datetime.datetime.now()

history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05, shuffle=True).history

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))



df_history = pd.DataFrame(history)

#model evaluate
score = model.evaluate(X_test, X_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#prediction
predictions = model.predict(X_test)


#plotting

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)


mx = round(max(scored['Loss_mae']),2)

th = round(((mx * 10)/100) + mx,3)


X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = th
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()


scored['Anomaly'].value_counts()

scored.tail()


model.save("tf-k-CNN_model.h5")
print("Model saved")


#plot loss anomaly vs loss no_anomaly

loss_anomaly_df =scored.loc[scored["Anomaly"]==True]
loss_noanomaly_df =scored.loc[scored["Anomaly"]==False]

loss_anomaly =loss_anomaly_df['Loss_mae']
loss_noanomaly =loss_noanomaly_df['Loss_mae']

import matplotlib.pyplot as plt
%matplotlib inline

#Training vs Validation Loss Plot
loss_anomaly =loss_anomaly_df['Loss_mae']
loss_noanomaly =loss_noanomaly_df['Loss_mae']

plt.figure()
plt.plot(loss_anomaly, 'b', label='loss_anomaly', color="red")
plt.plot(loss_noanomaly, 'b', label='loss_noanomaly', color="blue")

plt.title('LOSS-ANOMALY vs LOSS-NO_ANOMALY')
plt.legend()
plt.xlabel('index')
plt.ylabel('LOSS-ANOMALY and LOSS-NO_ANOMALY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1a.png')
plt.show()





#marplotlib
#loss vs val_loss

import matplotlib.pyplot as plt

#Training vs Validation Loss Plot
loss = history['loss']
val_loss = history['val_loss']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, loss, 'b', label='loss', color="red")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('LOSS vs VALIDATION LOSS')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('LOSS vs VALIDATION LOSS')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1b.png')
plt.show()


#accuracy vs val_accuracy

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
val_acc = history['val_acc']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'b', label='accuracy', color="red")
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('ACCURACY vs VALIDATION ACCURACY')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs VALIDATION ACCURACY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1c.png')
plt.show()


#mse vs vmse

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
mse = history['mean_squared_error']
vmse = history['val_mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, mse, 'b', label='mean_squared_error', color="red")
plt.plot(epochs, vmse, 'b', label='val_mean_squared_error')
plt.title('MSE vs VMSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE vs VMSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1d.png')
plt.show()

#mae vs vmae

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
mae = history['mean_absolute_error']
vmae = history['val_mean_absolute_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, mae, 'b', label='mean_absolute_error', color="red")
plt.plot(epochs, vmae, 'b', label='val_mean_absolute_error')
plt.title('MAE vs VMAE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MAE vs VMAE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1e.png')
plt.show()

#loss vs mse

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
loss = history['loss']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, loss, 'b', label='loss', color="red")
plt.plot(epochs, mse, 'b', label='mean_squared_error')
plt.title('LOSS vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('LOSS vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1f.png')
plt.show()



#loss vs mae
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
loss = history['loss']
mae = history['mean_absolute_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, loss, 'b', label='loss', color="red")
plt.plot(epochs, mae, 'b', label='mean_absolute_error')
plt.title('LOSS vs MAE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('LOSS vs MAE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1g.png')
plt.show()
#loss vs accuracy

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'b', label='Training acc', color="red")
plt.plot(epochs, val_acc, 'b', label='Validation acc', color="green")
plt.plot(epochs, loss, 'b', label='Training loss', color="blue")
plt.plot(epochs, val_loss, 'b', label='Validation loss', color="black")
plt.title('LOSS vs ACCURACY vs VLOSS vs VACCURACY')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('LOSS vs ACCURACY vs VLOSS vs VACCURACY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1h.png')
plt.show()

#loss vs accuarcy

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'b', label='accuracy', color="red")
plt.plot(epochs, loss, 'b', label="loss")
plt.title('LOSS vs ACCURACY')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('LOSS vs ACCURACY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1i.png')
plt.show()

               

#accuracy vs mae
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mae = history['mean_absolute_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'b', label='accuracy', color="red")
plt.plot(epochs, mae, 'b', label="mean_absolute_error")
plt.title('ACCURACY vs MAE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MAE vs ACCURACY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1j.png')
plt.show()

#acc vs mae


import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'b', label='accuracy', color="red")
plt.plot(epochs, mse, 'b', label="mean_squared_error")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE vs ACCURACY')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\1k.png')
plt.show()


#----------------
#dot circle plot------------1111-----------

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'ob', label='accuracy', color="red")
plt.plot(epochs, mse, 'or', label="mean_squared_error", color="blue")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\2a.png')
plt.show()


#dashed line plot-----22222------------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc,  label='accuracy', color ='red', linewidth=2, markersize=5, linestyle='dashed')
plt.plot(epochs, mse, label="mean_squared_error",color='green',  linestyle='dashed',
linewidth=2, markersize=12)
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\3a.png')
plt.show()

#plus plot------333333333333---------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'r+', label='accuracy', color="red")
plt.plot(epochs, mse, 'r+', label="mean_squared_error", color="black")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\4a.png')
plt.show()

#triangle plot------444444444444444444444444444----------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'g^', label='accuracy', color="red")
plt.plot(epochs, mse, 'g^', label="mean_squared_error", color="blue")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\5a.png')
plt.show()


#square plot----------555555555555555-------------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'rs-', label='accuracy', color="red")
plt.plot(epochs, mse, 'rs-', label="mean_squared_error", color="blue")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\6a.png')
plt.show()

#line square plot --------66666666666666666------------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.plot(epochs, acc, 'rs', label='accuracy', color="red")
plt.plot(epochs, mse, 'rs', label="mean_squared_error", color="blue")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\7a.png')
plt.show()

#bar plot

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.bar(epochs, acc,  label='accuracy', color="red", width = 2)
plt.bar(epochs, mse,  label="mean_squared_error", color="blue", width = 2)
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\8a.png')
plt.show()


#scatter plot with bar
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.scatter(epochs, acc,  label='accuracy', color="red")
plt.bar(epochs, mse,  label="mean_squared_error", color="blue")
plt.title('ACCURACY vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\9a.png')
plt.show()

#accuracy bar plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
#mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.bar(epochs, acc,  label='accuracy', color="red", width = 2)
#plt.bar(epochs, mse,  label="mean_squared_error", color="blue", width = 2)

plt.title('ACCURACY')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\10a.png')
plt.show()

#MSE bar plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
#acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
#plt.bar(epochs, acc,  label='accuracy', color="red", width = 2)
plt.bar(epochs, mse,  label="mean_squared_error", color="blue", width = 2)

plt.title('MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\11a.png')
plt.show()


#bar Plot for acc, loss, mae, mse
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
#mae = history['mean_absolute_error']
epochs = range(nb_epochs)
plt.figure()
plt.bar(epochs, acc,  label='accuracy', color="red")
plt.bar(epochs, loss,  label='loss', color="green")
#plt.bar(epochs, mae,  label='mean_absolute_error', color="red")
plt.bar(epochs, mse,  label="mean_squared_error", color="blue")

plt.title('ACCURACY vs MSE vs LOSS')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy, loss, mse')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\12a.png')
plt.show()

#SCATTER Plot for acc, loss, mae, mse

import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']

#mae = history['mean_absolute_error']
epochs = range(nb_epochs)
plt.figure()
plt.scatter(epochs, acc,  label='accuracy', color="blue")
plt.scatter(epochs, loss,  label='loss', color="green")
#plt.scatter(epochs, mae,  label='mean_absolute_error', color="red")
plt.scatter(epochs, mse,  label="mean_squared_error", color="yellow")

plt.title('ACCURACY vs MSE vs LOSS')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy, loss, mse')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\13a.png')
plt.show()

#area chart--------8888888888888888888-----------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
mse = history['mean_squared_error']
epochs = range(nb_epochs)
plt.figure()
plt.fill_between(epochs, acc,  label='accuracy', color="skyblue")
plt.plot(epochs, acc,   color="blue")

plt.fill_between(epochs, mse,  label='mse', color="green")
plt.plot(epochs, mse,   color="red")

plt.title('ACCURACY vs loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy and loss')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\14a.png')
plt.show()


#area chart--------99999999999999999-----------all--------
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.fill_between(epochs, acc,  label='accuracy', color="skyblue")
plt.plot(epochs, acc,   color="blue")

#plt.fill_between(epochs, mse,  label='mse', color="green")
#plt.plot(epochs, mse,   color="red")


plt.fill_between(epochs, loss,  label='loss', color="pink")
plt.plot(epochs, loss,   color="black")

plt.fill_between(epochs, mse,  label='mae', color="yellow")
plt.plot(epochs, mse,   color="grey")


plt.title('ACCURACY vs loss vs MAE vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs loss vs MAE vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\15a.png')
plt.show()


#sns pairplot all result parameter
# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt

hst =pd.DataFrame(history)

# Basic correlogram
sns.pairplot(hst)
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\16a.png')
plt.show()

#sns heatmap all result parameter
# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt

hst =pd.DataFrame(history)

# Basic correlogram
sns.heatmap(hst)
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\17a.png')
plt.show()

"""
import seaborn as sns
import matplotlib.pyplot as plt

hst =pd.DataFrame(history)

hst.plot(kind='pie', subplots=True, figsize=(8, 8))
"""


#hist plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
#acc = history['acc']
loss = history['loss']
epochs = range(nb_epochs)
plt.figure()
plt.hist(loss,  label="loss", rwidth=0.9,alpha=0.3,color='blue',bins=15,edgecolor='red')
# Defines X and Y axis labels
plt.xlabel('epoch')
plt.ylabel('loss')

plt.title('loss')
plt.legend()
#save and display the plot 
#plt.savefig('C:\\Users\\Dell\\Desktop\\AV Plotting images\\matplotlib_plotting_10.png',dpi=300,bbox_inches='tight') 
plt.show();

#or

import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt 
    
n_bins = 50
x  = history['acc']
 
    
colors = ['green']
  
plt.hist(x, n_bins, density = True,  
         histtype ='bar', 
         color = colors, 
         label = colors) 
  
plt.legend(prop ={'size': 10}) 
  
plt.title('loss hist plt', 
          fontweight ="bold") 
  
plt.show() 


#--------------------------------------------
#line dashed plot 1 plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.plot(epochs, acc,  label ="accurcay", color="red", linestyle=':' , linewidth=4)

plt.plot(epochs, mse,   label ="mse", color="green", linestyle='-.' , linewidth=4)

plt.plot(epochs, loss, label ="loss",  color="blue", linestyle='-' , linewidth=4)

#plt.plot(epochs, mae,  label ="mae", color="blue", linestyle='-' , linewidth=4)


plt.title('ACCURACY vs loss vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs loss vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\18a.png')
plt.show()



#line dashed 2 plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.plot(epochs, acc,  label ="accuracy", color="red", linestyle=':' , linewidth=4)

plt.plot(epochs, mse,   label ="mse", color="green", linestyle='-.' , linewidth=4)

plt.plot(epochs, loss,   label ="loss", color="blue", linestyle='--' , linewidth=4)

#plt.plot(epochs, mae,  label ="mae",  color="blue", linestyle='--' , linewidth=4)


plt.title('ACCURACY vs LOSS vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs LOSS vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\19a.png')
plt.show()

#dot plot
import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.plot(epochs, acc,  label = "accuracy",color="red", linestyle=':' , linewidth=4)

plt.plot(epochs, mse,   label ="mse", color="green", linestyle=':' , linewidth=4)

plt.plot(epochs, loss,   label ="loss", color="blue", linestyle=':' , linewidth=4)

#plt.plot(epochs, mae,   label ="mae", color="blue", linestyle=':' , linewidth=4)


plt.title('ACCURACY vs LOSS vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs LOSS vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\20a.png')
plt.show()



import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.plot(epochs, acc,  label ='accuracy',color="red", linestyle='-.' , linewidth=3)

plt.plot(epochs, mse,  label ="MSE", color="green", linestyle='-.' , linewidth=3)

plt.plot(epochs, loss,  label ="LOSS", color="blue", linestyle='-.' , linewidth=3)

#plt.plot(epochs, mae,   label ="MAE", color="blue", linestyle='-.' , linewidth=3)


plt.title('ACCURACY vs LOSS vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs LOSS vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\21a.png')
plt.show()




import matplotlib.pyplot as plt
#Training vs Validation Loss Plot
acc = history['acc']
loss = history['loss']
mse = history['mean_squared_error']
mae = history['mean_absolute_error']

epochs = range(nb_epochs)
plt.figure()

plt.plot(epochs, acc,  label ='accuracy',color="red", marker='o', markerfacecolor='black', markersize=5, linewidth=3)


plt.plot(epochs, mse,  label ="MSE", color="green", marker='o', markerfacecolor='pink', markersize=5, linewidth=3)


#plt.plot(epochs, mae,   label ="mae", color="black", linestyle='--' , linewidth=3)

plt.plot(epochs, loss,   label ="loss", color="blue", marker='o', markerfacecolor='yellow', markersize=5, linewidth=3)



plt.title('ACCURACY vs LOSS vs MSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('ACCURACY vs LOSS vs MSE')
plt.savefig(r'C:\Users\hp\Desktop\priyasoftweb\STM32\sierra project\cnn_lstm-xai-output\vis\22a.png')
plt.show()