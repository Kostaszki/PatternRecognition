import numpy as np
import tensorflow as tf
import keras
import json
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from keras.models import model_from_json


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
                    dataMatrix[j][i] = value
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

    train_x = np.array(training_DataPoints)
    val_x = np.array(val_DataPoints)
    print("Length of training data set: {:f}".format(len(train_x)))
    print("Length of validation data set: {:f}".format(len(train_y)))
    return train_x, train_y, val_x, val_y

def load_model():
    # load json and create model
    json_file = open("../nets/net_LSTM.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("../nets/net_LSTM_weights.h5")
    print("Loaded model from disk")
    return model

def create_FT_model(num_hidden_units = 50, lr = 0.001, activation = 'tanh', epsilon = 1e-8):
    model = load_model()
    opt = Adam(lr = lr, epsilon = epsilon)
                
    FT_model = model

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return FT_model

def train(epochs = 50, cross_validation = False):
    filepath = "../training_data/fine_tuning_dataset.mat"

    if cross_validation:
        seed = 7
        cvscores = []
        train_x, train_y, val_x, val_y = load_dataset(filepath, 1)
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for train, test in kfold.split(train_x, train_y):
            model = create_FT_model(lr = 0.01)
            history = model.fit(train_x[train], train_y[train], epochs=epochs, batch_size=32, verbose=1)
            # evaluate the model
            scores = model.evaluate(train_x[test], train_y[test], verbose=1)
            print("%s: %.2f / %s: %.2f%%" % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]*100))
            cvscores.append((scores[0], scores[1] * 100))
        print("%.2f (+/- %.2f) / %.2f%% (+/- %.2f%%)" % (np.mean([l[0] for l in cvscores]), np.std([l[0] for l in cvscores]), np.mean([l[1] for l in cvscores]), np.std([l[1] for l in cvscores])) )       
    else:
        train_x, train_y, val_x, val_y = load_dataset(filepath)
        model = create_FT_model(lr = 0.01)
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=32, validation_data=(val_x, val_y))
    return history, model

def save_net(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("net_{}.json".format(name), "w") as json_file:
        json_file.write(model_json)               
    # serialize weights to HDF5
    model.save_weights("net_{}_weights.h5".format(name))

def compute_accuracy(labels, predictions):
    # True positives and negatives (TPN)
    TPN = 0
    for i in range(0,len(predictions)):
        if labels[i] - predictions[i] == 0:
            TPN = TPN + 1
    accuracy = TPN/len(predictions)
    print("Accuracy: {:f}".format(accuracy))

def compute_precision(labels, predictions):
    #True positives (TP), False positives (FP)
    TP = 0
    FP = 0
    for i in range(0,len(predictions)):
        if labels[i] == 1 and predictions[i] == 1:
            TP = TP + 1
        elif labels[i] == 0 and predictions[i] == 1:
            FP = FP + 1
    precision = TP/(TP + FP)
    print("Precision: {:f}".format(precision))

def compute_recall(labels, predictions):
    # True Positives (TP), False Negatives (FN)
    TP = 0
    FN = 0
    for i in range(0,len(predictions)):
        if labels[i] == 1:
            if predictions[i] == 1:
                TP = TP + 1
            else:
                FN = FN + 1
    recall = TP/(TP + FN)
    print("Recall: {:f}".format(recall))

def test_model(model):
    filepath = 'data/training_samples_10k_reg2.mat'
    test_x, test_y, val_x, val_y = load_dataset(filepath, 1)
                
    test_predict = model.predict_classes(test_x, 128)
                        
    # Compute Recall, Accuracy and Precision
    compute_recall(test_y, test_predict)
    compute_accuracy(test_y, test_predict)
    compute_precision(test_y, test_predict)

def save_history(history, name):
    with open("{}.json".format(name), 'w') as f:
        json.dump(history.history, f)


history, model = train(epochs = 50, cross_validation = False)
save_history(history, "history_FT_LSTM")
save_net(model, "FT_LSTM")
