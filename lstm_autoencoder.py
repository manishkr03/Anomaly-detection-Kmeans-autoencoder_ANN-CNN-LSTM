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
    
    lstm1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(input_layer)
    
    output_layer = TimeDistributed(Dense(X.shape[2]))(lstm1)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

model = autoencoder_model(X_train)
model.compile(optimizer="adam", loss='mae', metrics=["accuracy", 
          "mean_squared_error", "mean_absolute_error"])

model.summary()


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


#train

xyz =[[0.24593154, 0.80254674, 0.22283804, 0.7602443],
[-0.06385729,  1.0165949 , -0.00471553,  0.98914695]]


xyz_df =pd.DataFrame(xyz)

xyz_df =xyz_df.to_numpy()
#scaled the data
xyz_test = xyz_df.reshape(xyz_df.shape[0], 1, xyz_df.shape[1])
print("Training data shape:", xyz_test.shape)

pred_xyz =model.predict(xyz_test)


from tensorflow.keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


pred_xyz =model.predict(xyz_test)









