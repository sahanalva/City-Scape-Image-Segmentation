
import numpy as np 
import os
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import utils

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def unet(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inp)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inp, outputs = conv10)

    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def segnet(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):

    inp = Input(input_size)
    # Encoder
    x = Convolution2D(64, 3, padding = "same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(num_classes, 1, padding = "valid")(x)
    x = Activation("softmax")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def resnet_concat(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inp)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1_2 = concatenate([conv1_2, conv1], axis = 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2_2 = concatenate([conv2_2, conv2], axis = 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3_2 = concatenate([conv3_2, conv3], axis = 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4_2 = concatenate([conv4_2, conv4], axis = 3)
    drop4 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(up6)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
    conv6_2 = concatenate([conv6_2, conv6], axis = 3)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6_2))
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(up7)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7_2 = concatenate([conv7_2, conv7], axis = 3)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7_2))
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(up8)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8_2 = concatenate([conv8_2, conv8], axis = 3)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8_2))
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(up9)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9_2 = concatenate([conv9_2, conv9], axis = 3)

    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9_2)

    model = Model(inputs = inp, outputs = conv10)

    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def resnet_simple(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inp)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1_2 = Add()([conv1_2, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2_2 = Add()([conv2_2, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3_2 = Add()([conv3_2, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4_2 = Add()([conv4_2, conv4])
    drop4 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(up6)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
    conv6_2 = Add()([conv6_2, conv6])

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6_2))
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(up7)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7_2 = Add()([conv7_2, conv7])

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7_2))
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(up8)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8_2 = Add()([conv8_2, conv8])

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8_2))
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(up9)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9_2 = Add()([conv9_2, conv9])

    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9_2)

    model = Model(inputs = inp, outputs = conv10)

    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_resnet_concat(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inp)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1_2 = concatenate([conv1_2, conv1], axis = 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2_2 = concatenate([conv2_2, conv2], axis = 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3_2 = concatenate([conv3_2, conv3], axis = 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4_2 = concatenate([conv4_2, conv4], axis = 3)
    drop4 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
    conv6_2 = concatenate([conv6_2, conv6], axis = 3)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6_2))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7_2 = concatenate([conv7_2, conv7], axis = 3)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7_2))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8_2 = concatenate([conv8_2, conv8], axis = 3)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8_2))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9_2 = concatenate([conv9_2, conv9], axis = 3)

    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9_2)

    model = Model(inputs = inp, outputs = conv10)

    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_resnet_simple(pretrained_weights = None, num_classes = 20, input_size = (256,256,1)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inp)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1_2 = Add()([conv1_2, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2_2 = Add()([conv2_2, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3_2 = Add()([conv3_2, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4_2 = Add()([conv4_2, conv4])
    drop4 = Dropout(0.5)(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
    conv6_2 = Add()([conv6_2, conv6])

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6_2))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
    conv7_2 = Add()([conv7_2, conv7])

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7_2))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
    conv8_2 = Add()([conv8_2, conv8])

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8_2))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9_2 = Add()([conv9_2, conv9])

    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9_2)

    model = Model(inputs = inp, outputs = conv10)

    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def identity_block(input_x, filters):
    
    filters_1, filters_2, filters_3 = filters

    x = Conv2D(filters_1, (1, 1))(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters_2, 3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters_3, (1, 1))(x)
    x = BatchNormalization()(x)
    
    x = add([x, input_x])
    x = Activation('relu')(x)
    return x


def conv_block(input_x, filters, strides=(2, 2)):

    filters_1, filters_2, filters_3 = filters



    x = Conv2D(filters_1, (1, 1), strides=strides)(input_x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters_2, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters_3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters_3, (1, 1),
                      strides=strides)(input_x)
    shortcut = BatchNormalization()(shortcut)


    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_encoder(pretrained_weights = 'imagenet', num_classes = 20, input_size = (256,256,1)):

    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

    inp = Input(input_size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (7, 7),
               strides=(2, 2))(x)
    feature_map_1 = x

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    feature_map_2 = ZeroPadding2D((1, 1))(x)
    feature_map_2 = Lambda(lambda x: x[:, :-1, :-1, :])(feature_map_2)

    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    feature_map_3 = x

    x = conv_block(x, [256, 256, 1024])
    x = identity_block(x,  [256, 256, 1024])
    x = identity_block(x,  [256, 256, 1024])
    x = identity_block(x,  [256, 256, 1024])
    x = identity_block(x,  [256, 256, 1024])
    x = identity_block(x,  [256, 256, 1024])
    feature_map_4 = x

    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    feature_map_5 = x




    model = Model(inputs = inp, outputs = feature_map_5)


    if pretrained_weights == 'imagenet':
        weights_path = utils.get_file(
            pretrained_url.split("/")[-1], pretrained_url)
        model.load_weights(weights_path)

    return inp,[feature_map_1,feature_map_2,feature_map_3,feature_map_4,feature_map_5]

def resnet50_encoder_unet_decoder(pretrained_weights = 'None', num_classes = 20, input_size = (256,256,1)):
    inp, feature_map_list= resnet50_encoder(pretrained_weights = 'imagenet', num_classes = num_classes, input_size = input_size)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(feature_map_list[4]))
    merge6 = concatenate([feature_map_list[3],up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([feature_map_list[2],up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([feature_map_list[1],up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([feature_map_list[0],up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    
    up10 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv9))
    merge9 = concatenate([inp,up10], axis = 3)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv10)
    conv11 = Conv2D(20, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = inp, outputs = conv11)
    model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True), loss = 'categorical_crossentropy', metrics = ['accuracy',iou_coef])

    return model

