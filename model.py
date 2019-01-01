from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from scipy import ndimage
from random import randint

from keras.layers import Conv2D, MaxPooling2D, Dropout, LeakyReLU, Flatten, Dense
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import applications

WIDTH = 256
HEIGHT = 256

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
        print(output[0])
        draw_box(x[id], output[0], False)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def draw_box(img, mask, display=True):
    img = img * 255

    mask = mask.astype('int')
    img = cv2.rectangle(np.uint8(img), (mask[1], mask[0]), (mask[3], mask[2]), (255, 0, 0), 3, cv2.LINE_8)
    
    if display: 
        plt.imshow(img)
        plt.show()
    else:
        cv2.imwrite('img.png',img)

def l2_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return np.square(y_true_f - y_pred_f)

def l1_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)

def fill_raw(data_path):
    if LOAD:
        x = np.load('x.npy')
        y = np.load('y.npy')
        return x, y
    

    path, dirs, files = next(os.walk(data_path))

    count = int(len(next(os.walk(data_path))[2]))

    x_arr = np.zeros((count, WIDTH, HEIGHT, 3), dtype='int')
    y_arr = np.zeros((count, 4)) #(top_y, top_x, height, width)

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

            #using scipy since cv2 doesn't like gifs even tho these only have one frame
            #format is (y,x) for some reason

            img = ndimage.imread(data_path + '/../train_masks/' + name + '_mask.gif')
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA) 

            top_left = np.zeros((2), dtype='int') #(y,x)
            bottom_right = np.zeros((2), dtype='int') #(y,x)

            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if np.all(img[y][x]):
                        if top_left[0] == 0:
                            top_left[0] = y
                        else:
                            bottom_right[0] = y

            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    if np.all(img[y][x]):
                        if top_left[1] == 0:
                            top_left[1] = x    
                        else:
                            bottom_right[1] = x             
            
            height = bottom_right[0] - top_left[0]
            width = bottom_right[1] - top_left[1]

            y_arr[i] = np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]]) #(y1, x1, y2, x2)

            i += 1
                
    np.save('x', x_arr)
    np.save('y', y_arr)

    return x_arr, y_arr

x, y = fill_raw('./data/train_hq')
x = x / np.max(x)
 
base_model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (WIDTH, HEIGHT, 3))

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-5:]:
    layer.trainable = True

top = Conv2D(128, kernel_size=(3,3), activation='relu', strides=2, padding='same')(base_model.output)
top = Dropout(0.25)(top)

top = Conv2D(64, kernel_size=(3,3), activation='relu', strides=2, padding='same')(top)
top = Dropout(0.25)(top)

top = Conv2D(32, kernel_size=(3,3), activation='relu', strides=2, padding='same')(top)
top = Dropout(0.25)(top)

top = Conv2D(16, kernel_size=(3,3), activation='relu', strides=2, padding='same')(top)
top = Dropout(0.25)(top)


top = Flatten()(top)
predictions = Dense(4, activation='linear')(top)

model_final = Model(input=base_model.input, output=predictions)
model_final.compile(loss=l1_loss, optimizer=Adam(.0001))

checkpoint = ModelCheckpoint('data.hdf5', save_best_only=True, verbose=1, monitor='val_loss')
stopping = EarlyStopping(monitor='val_loss', min_delta=.0001, patience=15)


model_final.fit(x, y, validation_split=.4, callbacks=[Histories(x), checkpoint, stopping], epochs=500, batch_size=1)