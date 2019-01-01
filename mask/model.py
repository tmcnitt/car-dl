from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from scipy import ndimage
from random import randint

from keras.layers import merge, Conv2D, MaxPooling2D, Dropout, LeakyReLU, Flatten, Dense, Input, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import applications


WIDTH = 256
HEIGHT = 256
CHANNELS = 3

LOAD = False

class Histories(Callback):
    def __init__(self, x):
        self.x  = x
        
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        id = randint(0, x.shape[0]-1)
        img = x[id]
        img = np.reshape(img, (1, WIDTH, HEIGHT,3))

        output = self.model.predict(img)
        output = np.array((output > .5)).astype('int')
        output = np.reshape(output, (WIDTH, HEIGHT))

        cv2.imwrite('./logs/output.png', output)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def fill_raw(data_path):
    if LOAD:
        print('using cache')
        x = np.load('x.npy')
        y = np.load('y.npy')
        return x, y

    path, dirs, files = next(os.walk(data_path))

    count = int(len(next(os.walk(data_path))[2]))

    x_arr = np.zeros((count, WIDTH, HEIGHT, 3), dtype='int')
    y_arr = np.zeros((count, WIDTH, HEIGHT, 1))

    i = 0
    sys.stdout.flush()
    for path, subdirs, files in (os.walk(data_path)):
        data = np.array(files)
        
        for name in tqdm(data):
            fullpath = os.path.join(path, name)

            img = cv2.imread(fullpath)   
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
            img = np.reshape(img, (WIDTH, HEIGHT, 3))
            x_arr[i] = img


            name = fullpath.split('/')[-1].strip('.jpg')
            img = ndimage.imread(data_path + '/../train_masks/' + name + '_mask.gif')
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA) 
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #masks have 3 channels but only two colors

            img = np.reshape(img, (WIDTH, HEIGHT, 1))
            y_arr[i] = img

            i += 1
                
    np.save('x', x_arr)
    np.save('y', y_arr)

    return x_arr, y_arr

x, y = fill_raw('../data/train_hq')
x = x / np.max(x)
y = y / np.max(y)

inputs = Input((WIDTH, HEIGHT,CHANNELS))
conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

output = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(inputs=[inputs],outputs=[output])

checkpoint = ModelCheckpoint('data.hdf5', save_best_only=True, verbose=1, monitor='val_loss')
stopping = EarlyStopping(monitor='val_loss', min_delta=.0001, patience=15)

model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss, metrics=[dice_coef])
model.summary()
model.fit(x, y, validation_split=.2, callbacks=[Histories(x), checkpoint, stopping], epochs=500, batch_size=2)
