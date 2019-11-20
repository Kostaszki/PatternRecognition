from __future__ import division
import scipy.io as sio
import tensorflow as tf
import keras
from keras.models import model_from_json
import time

def load_model():# load json and create model
    json_file = open("../nets/net_LSTM.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("../nets/net_LSTM_weights.h5")

    print("Loaded model from disk")
    return model

def load_dataset(filepath):
    mat_contents = sio.loadmat(filepath)
    preds = []
    preds_x = mat_contents.get('sw_mag')

    print("Shape data Matrix: {:s}".format(str(preds_x.shape)))
    return preds_x



filepath = "../preprocessed_data/LSTM_preprocess_test4.mat"

model = load_model()
start = time.time()

pred_time = 0

preds_x = load_dataset(filepath)
print("Length of Preds: {:s}".format(str(preds_x.shape)))
start_pred = time.time()
proba = model.predict_proba(preds_x, 8192)
end_pred = time.time()
pred_time = pred_time + (end_pred - start_pred)


end = time.time()
print('Elapsed time for prediction only: {}'.format(pred_time))
print('Elapsed time in total: {}'.format( end - start))
sio.savemat('results_test3_LSTM.mat', {'proba':proba})

