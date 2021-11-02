import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import numpy as np
import scipy.io as sio
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

### Tensorflow 2.0 version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


### Custom
def loadMatlabData(filePath, fileName):
    fileDir = filePath + fileName
    contentsMat = sio.loadmat(fileDir)
    train_id = contentsMat['Train_ID']
    train_code = contentsMat['Train_Input']
    train_label = contentsMat['Train_Label']
    train_size = train_label.shape[0]
    train_label = train_label.squeeze()

    valid_id = contentsMat['Valid_ID']
    valid_code = contentsMat['Valid_Input']
    valid_label = contentsMat['Valid_Label']
    valid_size = valid_label.shape[0]
    valid_label = valid_label.squeeze()
    return train_id, train_code, train_label, train_size, valid_id, valid_code, valid_label, valid_size

def func_OversampleDatasets(input_code, input_label, input_id, input_column=46):
    str_sampling = 'not majority'
    samp_id, samp_label = RandomOverSampler(random_state=0, sampling_strategy=str_sampling).fit_sample(input_id, input_label)

    org_id = input_id[:, 0]
    new_data = np.zeros(shape=(samp_id.shape[0], input_column))
    for ii in range(samp_id.shape[0]):
        now_id = samp_id[ii][0]
        find_id = np.where(org_id == now_id)
        get_data = input_code[find_id]
        new_data[ii] = get_data
    new_data = np.asarray(new_data)
    return new_data, samp_label, samp_id

def func_ShuffleDatasets(input_code, input_label, input_id, input_size):
    idx_shuffle = list(range(input_size))
    # np.random.seed(42)
    np.random.shuffle(idx_shuffle)

    shuffle_id = input_id[idx_shuffle, :]
    shuffle_code = input_code[idx_shuffle, :]
    shuffle_label = input_label[idx_shuffle]
    return shuffle_id, shuffle_code, shuffle_label

def func_GetDatasets(filePath, fileName):
    ### Load Train & Test Datasets
    train_id, train_code, train_label, train_size, valid_id, valid_code, valid_label, valid_size = loadMatlabData(filePath, fileName)
    print('Train', train_code.shape, train_label.shape, train_id.shape, train_size)
    print('Valid', valid_code.shape, valid_label.shape, valid_id.shape, valid_size)

    over_train, over_label, over_id = func_OversampleDatasets(train_code, train_label, train_id, input_column)
    size_over = over_id.shape[0]
    over_id, over_train, over_label = func_ShuffleDatasets(over_train, over_label, over_id, size_over)
    print('Oversample Train', over_train.shape, over_label.shape, over_id.shape, size_over)

    ### Convert to types => float32
    code_dtypes = 'float64'
    code_over = over_train.astype(code_dtypes)
    code_valid = valid_code.astype(code_dtypes)

    ### Distribution of classes
    print('=========================================================================================================')
    aa, counts_train = np.unique(train_label, return_counts=True)
    print('Distribution of classes in Train dataset: ', aa, counts_train, np.sum(counts_train))
    aa, counts_over = np.unique(over_label, return_counts=True)
    print('Distribution of classes in Oversample Train dataset: ', aa, counts_over, np.sum(counts_over))
    aa, counts_valid = np.unique(valid_label, return_counts=True)
    print('Distribution of classes in Valid dataset: ', aa, counts_valid, np.sum(counts_valid))
    print('=========================================================================================================')
    return code_over, over_label, code_valid, valid_label

### Define models
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_train_begin(self, logs=None):
        self.arr_loss_train = []
        self.arr_loss_valid = []

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch: ', epoch, logs)
        self.arr_loss_train.append(logs['loss'])
        self.arr_loss_valid.append(logs['val_loss'])

        ### Save models
        self.model.save_weights(filepath=self.save_path + 'model_weight_' + str(epoch) + '.h5')

def define_model(input_size):
    input_data = Input(shape=(input_size, ))

    ### Model (Region 46)
    x = Dense(64, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer=tf.random_normal_initializer(stddev=0.01))(input_data)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(32, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=losses.binary_crossentropy)

    print('Model==============')
    model.summary()
    return model

### Setting the parameter
batch_size = 64   # 배치 크기
learning_rate = 5e-4
filePath = './'
SavePath = './Upload_Github_20211102/'

input_column = 46
fileName = SavePath + 'Dataset_mini.mat'

### Load Train & Test Datasets
data_over, label_over, data_valid, label_valid = func_GetDatasets(filePath, fileName)

### Callback
callback_list = [
    CustomCallback(save_path=SavePath)
]

### Training
model = define_model(input_size=input_column)
hist = model.fit(data_over, label_over, batch_size=batch_size, validation_data=(data_valid, label_valid), epochs=100, verbose=0, callbacks=callback_list)
print('Train Loss: ', hist.history['loss'])
print('Valid Loss: ', hist.history['val_loss'])


