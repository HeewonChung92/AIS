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
    test_data = contentsMat['Test_Input']
    test_label = contentsMat['Test_Label']
    test_size = test_label.shape[0]
    test_label = test_label.squeeze()
    return test_data, test_label, test_size

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
    test_data, test_label, test_size = loadMatlabData(filePath, fileName)
    print('Test', test_data.shape, test_label.shape, test_size)

    ### Convert to types => float32
    code_dtypes = 'float64'
    code_test = test_data.astype(code_dtypes)

    ### Distribution of classes
    print('=========================================================================================================')
    aa, counts_test = np.unique(test_label, return_counts=True)
    print('Distribution of classes in Test dataset: ', aa, counts_test, np.sum(counts_test))
    print('=========================================================================================================')
    return code_test, test_label

### Define models
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
fileName = SavePath + 'TestDataset.mat'

### Load Train & Test Datasets
data_test, label_test = func_GetDatasets(filePath, fileName)

### Predict
model = define_model(input_size=input_column)
model.load_weights(SavePath + 'Model_Region46.h5')

test_pred = model.predict(data_test)
sio.savemat(SavePath + 'Result_DNN_Region46.mat', {'test_true': label_test, 'test_pred': test_pred})



