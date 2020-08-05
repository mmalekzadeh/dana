import numpy as np
import pandas as pd
from sklearn.utils import class_weight

def load_data(features=[0,1,2,3,4,5,6,7,8], location="wirst" , shuffle=True, data_path="../dana/datasets/Utwente/"):        

    act_lbls = ["walk", "stand", "jog", "sit", "bike", "ups", "downs",
                       "type", "write", "coffee", "talk", "smoke", "eat"]
    columns_labels = np.array(["timeStamp", "accX", "accY", "accZ", "linX", "linY","linZ",
                            "gyrX", "gyrY", "gyrZ", "magX", "magY", "magZ", "actLabel"])
    dataset = pd.DataFrame(columns = columns_labels)
    if location == "wirst":
        tmp = pd.read_csv(data_path+"smartphoneatwrist.csv", header=None)
    elif location == "pocket":
        tmp = pd.read_csv(data_path+"smartphoneatpocket.csv", header=None)
    else:
        raise Exception("Location <{}> is not defiend!".format(location))  
    tmp.columns = columns_labels
    dataset = dataset.append(tmp)
    dataset = dataset.drop(columns=['timeStamp'])
    dataset['actLabel'] -= 11111

    sensors = ["accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ","magX", "magY", "magZ"]
    sensors_and_label = sensors[:]
    sensors_and_label.append("actLabel")
    dataset  = dataset[sensors_and_label]
    dataset["actLabel"] = dataset["actLabel"].astype("category").cat.codes

    data_train = pd.DataFrame(columns = sensors_and_label)
    data_test = pd.DataFrame(columns = sensors_and_label)

    train_test_ratio = .80
    segment = 5
    start = 0
    step = (int)(((len(dataset)//len(act_lbls) ) // segment))
    end = step
    cut = (int)(train_test_ratio*(end-start))
    for i,_ in enumerate(act_lbls):
        for j in range(segment): 
            data_train = data_train.append(dataset[start:cut])  
            data_test = data_test.append(dataset[cut:end]) 
            start = end
            end += step
            cut = start+(int)(train_test_ratio*(end-start))

    data_train.shape, data_test.shape

    X_train = data_train[sensors].values
    X_test = data_test[sensors].values
    Y_train = data_train["actLabel"].values
    Y_test = data_test["actLabel"].values

    mean = X_train.mean(0)
    std  = X_train.std(0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    def windowing(X,Y, W,S):
        XX = []
        YY = []
        for i in range(0,(X.shape[0] - W),S):
            if Y[i] != Y[i+W-1]: 
                continue
            XX.append(X[i:i+W,:])
            YY.append(Y[i])
        return np.array(XX), np.array(YY)
    W = 128
    S = 64
    X_train, Y_train = windowing(X_train, Y_train, W, S)
    X_test, Y_test = windowing(X_test, Y_test, W, S)

    X_train = X_train[:,:,features]
    X_test = X_test[:,:,features]

    if shuffle:
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        Y_train = Y_train[indices] 


    act_weights = class_weight.compute_class_weight('balanced',range(len(act_lbls)),Y_train)
    act_weights = act_weights.round(4)
    act_weights = dict(zip(range(len(act_weights)),act_weights))

    del dataset, data_train, data_test
    return (X_train.astype("float32"), Y_train.astype("float32"), X_test.astype("float32"), Y_test.astype("float32"), act_lbls, act_weights)