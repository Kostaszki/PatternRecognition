from __future__ import division
import numpy as np
import time
import json
import keras
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold



# Load data and split into training and validation data set
def load_dataset(filepath, val_ratio = 0.8):
    mat_contents = sio.loadmat(filepath)
    training_DataPoints = []
    val_DataPoints = []
    training_label = []
    val_label = []
    cat = mat_contents.get('cat')
    samples_len = len(mat_contents.get('samples'))
    num_sample = 0
    for timeSeries in mat_contents.get('samples'):
        sizeN = len(timeSeries[0][0])
        dataMatrix = np.zeros(shape=(sizeN,3))
        for features in timeSeries:
            i = 0
            for feature in features:
                j = 0
                for value in feature:
                    dataMatrix[j][i] = np.float32(value)
                    j = j + 1
                i = i + 1 
        if num_sample < val_ratio*samples_len:
            training_DataPoints.append(dataMatrix)    
            training_label.append(cat[num_sample])
        else:
            val_DataPoints.append(dataMatrix)
            val_label.append(cat[num_sample])
        num_sample = num_sample + 1

    # Use labelencoder 
    labelencoder_y_1 = LabelEncoder()
    train_y = labelencoder_y_1.fit_transform(training_label)
    val_y = labelencoder_y_1.fit_transform(val_label)


    train_x = np.array(training_DataPoints, dtype="float32")
    val_x = np.array(val_DataPoints, dtype="float32")


    print("Length of training data set: {:f}".format(len(train_x)))
    print("Length of validation data set: {:f}".format(len(val_x)))
    return train_x, train_y, val_x, val_y
    
    
def create_model(num_hidden_units = 10, lr = 0.01, activation = 'tanh', dropout = 0.2, r_dropout = 0.2):
    # Specify training parameters
    opt = Adam(lr=lr)
    
    model = Sequential()
    model.add(LSTM(num_hidden_units, activation=activation, dropout = dropout, recurrent_dropout=r_dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train(epochs = 50, cross_validation = False):
    filepath = "../training_data/training_dataset.mat"
  
    if cross_validation:
        seed = 7
        cvscores = []
        train_x, train_y, val_x, val_y = load_dataset(filepath, 1)
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for train, test in kfold.split(train_x, train_y):
            model = create_model(lr = 0.01)
            model.fit(train_x[train], train_y[train], epochs=epochs, batch_size=1024, verbose=1)
            # evaluate the model
            scores = model.evaluate(train_x[test], train_y[test], verbose=1)
            print("%s: %.2f / %s: %.2f%%" % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]*100))
            cvscores.append((scores[0], scores[1] * 100))
        print("%.2f (+/- %.2f) / %.2f%% (+/- %.2f%%)" % (np.mean([l[0] for l in cvscores]), np.std([l[0] for l in cvscores]), np.mean([l[1] for l in cvscores]), np.std([l[1] for l in cvscores])) )            
        history = 0
    else:
        train_x, train_y, val_x, val_y = load_dataset(filepath)
        model = create_model(num_hidden_units = 10, lr = 0.005)
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=1024, validation_data=(val_x, val_y), verbose=1)
    
    return history, model
    
def save_net(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("net_{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("net_{}_weights.h5".format(name))


def save_history(history, name):
    with open("{}.json".format(name), 'w') as f:
        json.dump(history.history, f)


def compute_accuracy(labels, predictions):
    # True positives and negatives (TPN)
    TPN = 0
    for i in range(0,len(predictions)):
        if labels[i] - int(predictions[i]) == 0:
            TPN = TPN + 1
    accuracy = TPN/len(predictions)
    print("Accuracy: {:f}".format(accuracy))

def compute_precision(labels, predictions):
    # True positives (TP)
    # False positives (FP)
    TP = 0
    FP = 0
    for i in range(0,len(predictions)):
        if labels[i] == 1 and int(predictions[i]) == 1:
            TP = TP + 1
        elif labels[i] == 0 and int(predictions[i]) == 1:
            FP = FP + 1
    precision = TP/(TP + FP)
    print("Precision: {:f}".format(precision))

def compute_recall(labels, predictions):
    # True Positives (TP) and False Negatives (FN)
    TP = 0
    FN = 0
    for i in range(0,len(predictions)):
        if labels[i] == 1:
            if int(predictions[i]) == 1:
                TP = TP + 1
            else:
                FN = FN + 1
    recall = float(TP)/float(TP + FN)
    print("Recall: {:f}".format(recall))

def test_model(model):
    filepath = "../training_data/test_dataset.mat"
    test_x, test_y, val_x, val_y = load_dataset(filepath, 1)

    test_predict = model.predict_classes(test_x, 64)
                            
    # Compute Recall, Accuracy and Precision
    compute_recall(test_y, test_predict)
    compute_accuracy(test_y, test_predict)
    compute_precision(test_y, test_predict)

start = time.time()
history, model = train(epochs = 20, cross_validation = False)
end = time.time()
#save_net(model, "LSTM")
#save_history(history, "history_LSTM")
test_model(model)
print('Elapsed time: {}'.format( end - start))

