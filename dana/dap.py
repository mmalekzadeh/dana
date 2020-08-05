import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape

# Inspired by:
#          He, K., Zhang, X., Ren, S., & Sun, J. (2015). 
#          Spatial pyramid pooling in deep convolutional networks for visual recognition. 
#          IEEE transactions on pattern analysis and machine intelligence, 37(9), 1904-1916.    
#    Parts of the code is taken from this repo: https://github.com/yhenon/keras-spp
          
class DimensionAdaptivePooling(layers.Layer):
    """ Dimension Adaptive Pooling layer for 2D inputs.
    # Arguments
        pool_list: a tuple (W,H)
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each tuple in the list is the number of regions in that pool. For example [(8,6),(4,3)] would be 2
            regions with 1, 8x6 and 4x3 max pools, so 48+12 outputs per feature map.
        forRNN: binary
            Determines wheterh the layer after this is a recurrent layer (LSTM) or not (it is Dense)
        operation: string
            Either `max` or `avg`.
    # Input shape
        4D tensor with shape: `(samples, w, h, M)` .
    # Output shape
        2D or 3D tensor with shape: `(samples,  W*H*M)` or `(samples,  W, H*M)`.
    """
    def __init__(self, pooling_parameters, forRNN=False, operation="max", name=None, **kwargs):
        super(DimensionAdaptivePooling, self).__init__(name=name, **kwargs)
        self.pool_list = np.array(pooling_parameters)
        self.forRNN = forRNN
        self.W = self.pool_list[0]
        self.H = self.pool_list[1]
        self.num_outputs_per_feature_map =  self.W * self.H
        if operation == "max":
            self.operation = tf.math.reduce_max
        elif operation == "avg":
            self.operation = tf.math.reduce_mean
       
    def build(self, input_shape):        
        self.M = input_shape[3]      

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()        
        if self.forRNN:
          return tensor_shape.TensorShape([input_shape[0], self.W, self.H * self.M])
        else:
          return tensor_shape.TensorShape([input_shape[0], self.W * self.H * self.M])

    def get_config(self):
        config = {'dap pooling parameters': self.pool_list, 'forRNN': self.forRNN}
        base_config = super(DimensionAdaptivePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    


class DimensionAdaptivePoolingForSensors(DimensionAdaptivePooling):
    def __init__(self, pooling_parameters, forRNN=False, operation="max", name=None, **kwargs):
        super(DimensionAdaptivePoolingForSensors, self).__init__(pooling_parameters=pooling_parameters, 
                                                                forRNN=forRNN, 
                                                                operation=operation,
                                                                name=name, **kwargs)      
    def call(self, xp, mask=None):
        xp_dtype = xp.dtype                
        input_shape = tf.shape(xp)
        wp = input_shape[1] ## This is the number of sample points in each time-window (w')
        hp = input_shape[2] ## This is the number of sensor channels (h')
        
        ## X'' = X'
        xpp = tf.identity(xp)
        ## A =  maximum([(H-h')/3],0)$
        try:
            A = tf.cast(tf.math.maximum(tf.math.ceil((self.H-hp)/3),0), dtype=xp_dtype)
            for ia in range(tf.cast(A, tf.int32)):
                xpp = tf.concat([xpp, xp],2) 
            ## X′′=X′′[0∶w′,0∶maximum(h',H)]                                                                                 
            xpp = xpp[:, :wp, :tf.math.maximum(hp,self.H), :]
        except:
            A = tf.Variable(0,dtype=xp_dtype)        
        ## pw = w'/W  and ph = h'/H
        p_w = tf.cast(wp / self.W, dtype=xp_dtype)
        p_h = tf.cast(hp / self.H, dtype=xp_dtype)
        ## Z' = {}
        Zp = []
        for iw in range(self.W):                
            for ih in range(self.H):
                r1 = tf.cast(tf.math.round(iw * p_w), tf.int32)
                r2 = tf.cast(tf.math.round((iw+1) * p_w), tf.int32)
                if A == 0:
                    c1 = tf.cast(tf.math.round(ih *p_h), tf.int32)
                    c2 = tf.cast(tf.math.round((ih+1)*p_h), tf.int32)
                else:
                    c1 = tf.cast(tf.math.round(ih * tf.math.floor((A+1)*p_h)), tf.int32)
                    c2 = tf.cast(tf.math.round((ih+1) * tf.math.floor((A+1)*p_h)), tf.int32)
                try:                                               
                    Zp.append(self.operation(xpp[:, r1:r2, c1:c2, :], axis=(1, 2)))
                except:
                    Zp = []
        Zp = tf.concat(Zp, axis=-1)
        if self.forRNN:
            Zp = tf.reshape(Zp,(input_shape[0], self.W, self.H * self.M))
        else:
            Zp = tf.reshape(Zp,(input_shape[0], self.W * self.H * self.M))
        return Zp

## For those who want to work with Images
class DimensionAdaptivePoolingForImages(DimensionAdaptivePooling):
    def __init__(self, pooling_parameters, forRNN=False, operation="max", name=None, **kwargs):
        super(DimensionAdaptivePoolingForImages, self).__init__(pooling_parameters=pooling_parameters, 
                                                                forRNN=forRNN, 
                                                                operation=operation,
                                                                name=name, **kwargs)      
    def call(self, xp, mask=None):
        xp_dtype = xp.dtype                
        input_shape = tf.shape(xp)
        wp = input_shape[1] ## This is the number of rows (width)
        hp = input_shape[2] ## This is the number of columns (height)
        
        ## X'' = X'
        xpp = tf.identity(xp)        
        xpp = tf.image.resize(xpp, (tf.math.maximum(wp,self.W), tf.math.maximum(hp,self.H)))        
        ## pw = w'/W  and ph = h'/H
        p_w = tf.cast(tf.math.maximum(wp,self.W) / self.W, dtype=xp_dtype)
        p_h = tf.cast(tf.math.maximum(hp,self.H) / self.H, dtype=xp_dtype)
        ## Z' = {}
        Zp = []
        for iw in range(self.W):                
            for ih in range(self.H):
                r1 = tf.cast(tf.math.round(iw * p_w), tf.int32)
                r2 = tf.cast(tf.math.round((iw+1) * p_w), tf.int32)                
                c1 = tf.cast(tf.math.round(ih *p_h), tf.int32)
                c2 = tf.cast(tf.math.round((ih+1)*p_h), tf.int32)                
                try:                                               
                    Zp.append(self.operation(xpp[:, r1:r2, c1:c2, :], axis=(1, 2)))
                except:
                    Zp = []
        Zp = tf.concat(Zp, axis=-1)
        if self.forRNN:
            Zp = tf.reshape(Zp,(input_shape[0], self.W, self.H * self.M))
        else:
            Zp = tf.reshape(Zp,(input_shape[0], self.W * self.H * self.M))
        return Zp