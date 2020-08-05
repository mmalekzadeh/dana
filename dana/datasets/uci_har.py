import numpy as np
from sklearn.utils import class_weight

def load_data(features=[0,1,2,3,4,5], shuffle=True, data_path="../dana/datasets/UCIHAR/"):        
    X_train = np.load(data_path+"X_train.npy")
    Y_train = np.load(data_path+"y_train.npy").argmax(1)
    X_test = np.load(data_path+"X_test.npy")
    Y_test = np.load(data_path+"y_test.npy").argmax(1)        

    X_train = X_train[:,:,features]
    X_test = X_test[:,:,features]

    if shuffle:
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        Y_train = Y_train[indices]        

    act_lbls = ["WAL", "STU", "STN", "SIT", "STD", "LYI"]
    act_weights = class_weight.compute_class_weight('balanced',range(len(act_lbls)),Y_train)
    act_weights = act_weights.round(4)
    act_weights = dict(zip(range(len(act_weights)),act_weights))

    return (X_train, Y_train, X_test, Y_test, act_lbls, act_weights)