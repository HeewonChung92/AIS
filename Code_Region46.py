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

    ###============= Load Matlab files
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

    test_id = contentsMat['Test_ID']
    test_code = contentsMat['Test_Input']
    test_label = contentsMat['Test_Label']
    test_size = test_label.shape[0]
    test_label = test_label.squeeze()
    return train_id, train_code, train_label, train_size, valid_id, valid_code, valid_label, valid_size, test_id, test_code, test_label, test_size

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

def func_GetDatasets(filePath, fileName, input_column=6):
    ### Load Train & Test Datasets
    train_id, train_code, train_label, train_size, valid_id, valid_code, valid_label, valid_size, test_id, test_code, test_label, test_size = loadMatlabData(filePath, fileName)
    print('Train', train_code.shape, train_label.shape, train_id.shape, train_size)
    print('Valid', valid_code.shape, valid_label.shape, valid_id.shape, valid_size)
    print('Test', test_code.shape, test_label.shape, test_id.shape, test_size)

    over_train, over_label, over_id = func_OversampleDatasets(train_code, train_label, train_id, input_column)
    size_over = over_id.shape[0]
    over_id, over_train, over_label = func_ShuffleDatasets(over_train, over_label, over_id, size_over)
    print('Oversample Train', over_train.shape, over_label.shape, over_id.shape, size_over)

    ### Convert to types => float32
    code_dtypes = 'float64'
    code_over = over_train.astype(code_dtypes)
    code_valid = valid_code.astype(code_dtypes)
    code_test = test_code.astype(code_dtypes)

    ### Distribution of classes
    print('=========================================================================================================')
    aa, counts_train = np.unique(train_label, return_counts=True)
    print('Distribution of classes in Train dataset: ', aa, counts_train, np.sum(counts_train))
    aa, counts_over = np.unique(over_label, return_counts=True)
    print('Distribution of classes in Oversample Train dataset: ', aa, counts_over, np.sum(counts_over))
    aa, counts_valid = np.unique(valid_label, return_counts=True)
    print('Distribution of classes in Valid dataset: ', aa, counts_valid, np.sum(counts_valid))
    aa, counts_test = np.unique(test_label, return_counts=True)
    print('Distribution of classes in Test dataset: ', aa, counts_test, np.sum(counts_test))
    print('=========================================================================================================')
    return code_over, over_label, over_id, code_valid, valid_label, valid_id, code_test, test_label, test_id, counts_over, counts_valid, counts_test

def func_confusion_matrix(data, label, model):
    y_pred = model.predict(data)
    predicted = np.round(y_pred).squeeze()
    y_true = label
    cfs_mtx = confusion_matrix(y_true, predicted)

    total = np.sum(cfs_mtx)
    correct_tn = cfs_mtx[0][0]
    correct_fp = cfs_mtx[0][1]
    correct_fn = cfs_mtx[1][0]
    correct_tp = cfs_mtx[1][1]

    # vSensitivity, vSpecificity, vAccuracy = 0, 0, 0
    if correct_tp == 0:
        vSensitivity = 0
        vPPV = 0
    else:
        vSensitivity = 100 * (correct_tp / (correct_tp + correct_fn))
        vPPV = 100 * (correct_tp / (correct_tp + correct_fp))

    if correct_tn == 0:
        vSpecificity = 0
        vNPV = 0
    else:
        vSpecificity = 100 * (correct_tn / (correct_tn + correct_fp))
        vNPV = 100 * (correct_tn / (correct_tn + correct_fn))

    if correct_tp == 0 and correct_tn == 0:
        vAccuracy = 0
    else:
        vAccuracy = 100 * ((correct_tp + correct_tn) / total)

    return cfs_mtx, vSensitivity, vSpecificity, vAccuracy, total, vPPV, vNPV

def func_print_confusion_matrix(model, data_over, label_over, counts_over, data_valid, label_valid, counts_valid, data_test, label_test, counts_test):
    cfs_mtx_train, senTrain, specTrain, accTrain, totalTrain, ppvTrain, npvTrain = func_confusion_matrix(data_over, label_over, model)
    print('[Train], Sensitivity(TP/TP+FN): %2.4f,  Specificity(TN/TN+FP): %2.4f,  Accuracy(TP+TN/All): %2.4f,   '
          'PPV(TP/TP+FP): %2.4f,   NPV(TN/TN+FN): %2.4f   [TP:%d, TN:%d, FP:%d, FN:%d, All:%d(0-%d, 1-%d)]'
          % (senTrain, specTrain, accTrain, ppvTrain, npvTrain, cfs_mtx_train[1][1], cfs_mtx_train[0][0],
             cfs_mtx_train[0][1], cfs_mtx_train[1][0], totalTrain, counts_over[0], counts_over[1]))

    cfs_mtx_valid, senValid, specValid, accValid, totalValid, ppvValid, npvValid = func_confusion_matrix(data_valid, label_valid, model)
    print('[Valid], Sensitivity(TP/TP+FN): %2.4f,  Specificity(TN/TN+FP): %2.4f,  Accuracy(TP+TN/All): %2.4f,   '
          'PPV(TP/TP+FP): %2.4f,   NPV(TN/TN+FN): %2.4f   [TP:%d, TN:%d, FP:%d, FN:%d, All:%d(0-%d, 1-%d)]'
          % (senValid, specValid, accValid, ppvValid, npvValid, cfs_mtx_valid[1][1], cfs_mtx_valid[0][0],
             cfs_mtx_valid[0][1], cfs_mtx_valid[1][0], totalValid, counts_valid[0], counts_valid[1]))

    cfs_mtx_test, senTest, specTest, accTest, totalTest, ppvTest, npvTest = func_confusion_matrix(data_test, label_test, model)
    print('[Test], Sensitivity(TP/TP+FN): %2.4f,  Specificity(TN/TN+FP): %2.4f,  Accuracy(TP+TN/All): %2.4f,   '
          'PPV(TP/TP+FP): %2.4f,   NPV(TN/TN+FN): %2.4f   [TP:%d, TN:%d, FP:%d, FN:%d, All:%d(0-%d, 1-%d)]'
          % (senTest, specTest, accTest, ppvTest, npvTest, cfs_mtx_test[1][1], cfs_mtx_test[0][0], cfs_mtx_test[0][1],
             cfs_mtx_test[1][0], totalTest, counts_test[0], counts_test[1]))

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
fileName = SavePath + 'Dataset_Region46.mat'

### Load Train & Test Datasets
data_over, label_over, id_over, data_valid, label_valid, id_valid, data_test, label_test, id_test, counts_over, counts_valid, counts_test = func_GetDatasets(filePath, fileName, input_column=input_column)
input_over = data_over
input_valid = data_valid
input_test = data_test

model = define_model(input_size=input_column)
model.load_weights(SavePath + 'Model_Region46.h5')
func_print_confusion_matrix(model, input_over, label_over, counts_over, input_valid, label_valid, counts_valid, input_test, label_test, counts_test)
over_pred = model.predict(input_over, batch_size=batch_size)
valid_pred = model.predict(input_valid, batch_size=batch_size)
test_pred = model.predict(input_test)
sio.savemat(SavePath + 'Result_DNN_Region46.mat',
            {'over_true': label_over, 'over_pred': over_pred, 'valid_true': label_valid, 'valid_pred': valid_pred, 'test_true': label_test, 'test_pred': test_pred})



