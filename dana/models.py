import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from dana.dap import *

##################################################
def Ronao2016HumanOriginal(inp_shape, out_shape):    
    """
    @article{ronao2016human,
        title={Human activity recognition with smartphone sensors using deep learning neural networks},
        author={Ronao, Charissa Ann and Cho, Sung-Bae},
        journal={Expert systems with applications},
        volume={59},
        pages={235--244},
        year={2016},
        publisher={Elsevier}
    }
    """
    drp_out_dns = .8 
    nb_dense = 1000 
    kernel_regularizer = regularizers.l2(0.00005)

    inp = Input(inp_shape)
    x = Conv2D(96, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(inp)    
    x = MaxPool2D(pool_size=(3,1))(x)
    x = Conv2D(192, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(x)    
    x = MaxPool2D(pool_size=(3,1))(x)
    x = Conv2D(192, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(x) 
    x = MaxPool2D(pool_size=(3,1))(x)

    x = Flatten()(x)
    
    act = Dense(nb_dense, kernel_regularizer=kernel_regularizer,                 
                activation='relu', name="act_dns")(x)
    act = Dropout(drp_out_dns, name= "act_drp_out")(act)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(act)
    model = keras.models.Model(inputs=inp, outputs=out_act)

    return model


def Ronao2016HumanWithDAP(inp_shape, out_shape, pool_list=(4,6)):  
    drp_out_dns = .8 
    nb_dense = 1000 
    kernel_regularizer = regularizers.l2(0.00005)

    inp = Input(inp_shape)
    x = Conv2D(96, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(inp)        
    x = Conv2D(192, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(x)        
    x = Conv2D(192, kernel_size = (9,1), 
              kernel_regularizer=kernel_regularizer,
              strides=(1,1), padding='same', activation='relu')(x) 

    x = DimensionAdaptivePoolingForSensors(pool_list, operation="max", name ="DAP", forRNN=False)(x)    
  
    act = Dense(nb_dense, kernel_regularizer=kernel_regularizer,                 
                activation='relu', name="act_dns")(x)
    act = Dropout(drp_out_dns, name= "act_drp_out")(act)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(act)
    model = keras.models.Model(inputs=inp, outputs=out_act)

    return model

##################################################
def Ordonez2016DeepOriginal(inp_shape, out_shape):   
    """
    @article{ordonez2016deep,
        title={Deep convolutional and {LSTM} recurrent neural networks for multimodal wearable activity recognition},
        author={Ord{\'o}{\~n}ez, Francisco and Roggen, Daniel},
        journal={Sensors},
        volume={16},
        number={1},
        pages={115},
        year={2016},
        publisher={Multidisciplinary Digital Publishing Institute}
    }
    """   
    nb_filters = 64 
    drp_out_dns = .5 
    nb_dense = 128 
    
    inp = Input(inp_shape)

    x = Conv2D(nb_filters, kernel_size = (5,1),
              strides=(1,1), padding='valid', activation='relu')(inp)    
    x = Conv2D(nb_filters, kernel_size = (5,1),
              strides=(1,1), padding='valid', activation='relu')(x)
    x = Conv2D(nb_filters, kernel_size = (5,1), 
              strides=(1,1), padding='valid', activation='relu')(x)
    x = Conv2D(nb_filters, kernel_size = (5,1), 
              strides=(1,1), padding='valid', activation='relu')(x)    
    x = Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    act = LSTM(nb_dense, return_sequences=True, activation='tanh', name="lstm_1")(x)        
    act = Dropout(drp_out_dns, name= "dot_1")(act)
    act = LSTM(nb_dense, activation='tanh', name="lstm_2")(act)        
    act = Dropout(drp_out_dns, name= "dot_2")(act)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(act)

    model = keras.models.Model(inputs=inp, outputs=out_act)
    return model

def Ordonez2016DeepWithDAP(inp_shape, out_shape, pool_list=(4,6)):      
    nb_filters = 64 
    drp_out_dns = .5 
    nb_dense = 128 

    inp = Input(inp_shape)

    x = Conv2D(nb_filters, kernel_size = (5,1),
              strides=(1,1), padding='same', activation='relu')(inp)    
    x = Conv2D(nb_filters, kernel_size = (5,1),
              strides=(1,1), padding='same', activation='relu')(x)
    x = Conv2D(nb_filters, kernel_size = (5,1), 
              strides=(1,1), padding='same', activation='relu')(x)
    x = Conv2D(nb_filters, kernel_size = (5,1), 
              strides=(1,1), padding='same', activation='relu')(x)    
        
    x = DimensionAdaptivePoolingForSensors(pool_list, operation="max", name ="DAP", forRNN=True)(x)    

    act = LSTM(nb_dense, return_sequences=True, activation='tanh', name="lstm_1")(x)        
    act = Dropout(drp_out_dns, name= "dot_1")(act)
    act = LSTM(nb_dense, activation='tanh', name="lstm_2")(act)        
    act = Dropout(drp_out_dns, name= "dot_2")(act)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(act)

    model = keras.models.Model(inputs=inp, outputs=out_act)
    return model

##################################################
def Ignatov2018RealOriginal(inp_shape, out_shape):   
    """
    @article{ignatov2018real,
        title={Real-time human activity recognition from accelerometer data using Convolutional Neural Networks},
        author={Ignatov, Andrey},
        journal={Applied Soft Computing},
        volume={62},
        pages={915--922},
        year={2018},
        publisher={Elsevier}
    }
    """
    inp = Input(inp_shape)
    x = Conv2D(196, kernel_size = (16,6), strides=(1,6), padding='same', activation='relu')(inp)    
    x = MaxPool2D(pool_size=(4,1), padding="same")(x)
    x = Flatten()(x)        
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.95, name= "act_drp_out")(x)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(x)    
    model = keras.models.Model(inputs=inp, outputs=out_act)

    return model

def Ignatov2018RealWithDAP(inp_shape, out_shape, pool_list = (32,1)):   
    
    inp = Input(inp_shape)
    x = Conv2D(196, kernel_size = (16,6), strides=(1,6), padding='same', activation='relu')(inp)    
    x = MaxPool2D(pool_size=(4,1), padding="same")(x)
    x = DimensionAdaptivePoolingForSensors(pool_list, operation="max", name ="DAP", forRNN=False)(x)       
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.95, name= "act_drp_out")(x)
    out_act = Dense(out_shape, activation='softmax',  name="act_smx")(x)    
    model = keras.models.Model(inputs=inp, outputs=out_act)

    return model