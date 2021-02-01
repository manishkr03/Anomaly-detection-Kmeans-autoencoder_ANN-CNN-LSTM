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


data_dir = r"C:\Users\hp\Desktop\priyasoftweb\STM32\2nd_test"
merged_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
    
merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

# transform data file index to datetime and sort in chronological order
merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()
#merged_data.to_csv('Averaged_BearingTest_Dataset.csv')
print("Dataset shape:", merged_data.shape)
merged_data.head()



#merged_data.tail()    

train = merged_data['2004-02-12 10:52:39': '2004-02-15 12:52:39']
test = merged_data['2004-02-15 12:52:39':]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)


"""
from sklearn.model_selection import train_test_split

df = merged_data
RANDOM_SEED = 101

train, test = train_test_split(df, test_size=0.8, random_state = RANDOM_SEED)


print('Training data size   :', train.shape)
print('Validation data size :', test.shape)
"""

# normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train)
X_test_scaled = scaler.transform(test)


X_train_scaled.shape

#from tensorflow.keras.layers import Input, Dropout, LSTM , TimeDistributed , RepeatVector, Dense
#from tensorflow.keras.models import Model
#from tensorflow.keras import regularizers

"""
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Input, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from tensorflow.keras import regularizers
"""
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Input, TimeDistributed, Dropout
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from keras import regularizers


X_train = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
print("Test data shape:", X_test.shape)

"""
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Input, TimeDistributed, Dropout
from keras import regularizers
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Input
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

"""
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, activation='relu', kernel_size=4, padding='same', kernel_regularizer= regularizers.l2(0.00) )(inputs)
    L2 = Conv1D(4, activation='relu', kernel_size=4, padding='same')(L1)
    L4 = Conv1D(4, activation='relu', kernel_size=4, padding='same')(L2)
    L5 = Conv1D(16, activation='relu', kernel_size=4, padding='same')(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

model = autoencoder_model(X_train)
#model.compile(optimizer='adam', loss='mae', metrics=["accuracy"])
model.summary()


from keras import backend as K

def recall_m(X_test, pred):
    true_positives = K.sum(K.round(K.clip(X_test * pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(X_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(X_test, pred):
    true_positives = K.sum(K.round(K.clip(X_test * pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(X_test, pred):
    precision = precision_m(X_test, pred)
    recall = recall_m(X_test, pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
model.compile(optimizer='adam', loss='mae', metrics=['acc',f1_m,precision_m, recall_m])


nb_epochs = 100
batch_size = 10



history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history



df_history = pd.DataFrame(history)


predictions = model.predict(X_test)

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, predictions, verbose=1)

# serialize weights to HDF5
model.save_weights('model_cnn_weight.h5')

 
# serialize model to JSON
model_json = model.to_json()
with open("model_cnn_weights.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_cnn_weights.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model_cnn_weights.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_cnn_weights.h5")
print("Loaded model from disk")
 
#plotting


import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
# plt.xlim([0.0,.5])

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