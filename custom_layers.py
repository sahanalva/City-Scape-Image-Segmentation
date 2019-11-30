import keras.backend as K
from tensorflow.python.keras.layers import *


class MaxPoolingWithArgIndices(Layer):
    def __init__(self, pool_size,strides,padding='SAME',**kwargs):
        super(MaxPoolingWithArgIndices, self).__init__(**kwargs)
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        return
    def call(self,x):
        pool_size=[1,self.pool_size, self.pool_size,1]
        strides=[1,self.strides,self.strides,1]
        output1,output2=K.tf.nn.max_pool_with_argmax(x,pool_size,strides,self.padding)
        return [output1,output2]
                                                        
    def compute_output_shape(self, input_shape):

        output_shape=(input_shape[0],input_shape[1]//self.pool_size,input_shape[2]//self.pool_size,input_shape[3])
        return [output_shape,output_shape]


class UpSamplingUsingArgIndices(Layer):
    def __init__(self,indices, **kwargs):
        super(UpSamplingUsingArgIndices, self).__init__(**kwargs)
        self.indices = indices
        return
    def call(self,x):
        indices = self.indices
        argmax=K.cast(K.flatten(indices),'int32')
        max_value=K.flatten(x)
        input_shape=K.shape(x)
        with K.tf.compat.v1.variable_scope(self.name):
            batch_size=input_shape[0]
            image_size=input_shape[1]*input_shape[2]*input_shape[3]
            output_shape=[input_shape[0],input_shape[1]*2,input_shape[2]*2,input_shape[3]]
            indices_0=K.flatten(K.tf.matmul(K.reshape(K.tf.range(batch_size),(batch_size,1)),K.ones((1,image_size),dtype='int32')))
            indices_1=argmax%(image_size*4)//(output_shape[2]*output_shape[3])
            indices_2=argmax%(output_shape[2]*output_shape[3])//output_shape[3]
            indices_3=argmax%output_shape[3]
            indices=K.tf.stack([indices_0,indices_1,indices_2,indices_3])
            output=K.tf.scatter_nd(K.transpose(indices),max_value,output_shape)
            return output
    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1]*2,input_shape[2]*2,input_shape[3]